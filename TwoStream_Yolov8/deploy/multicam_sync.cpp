#include <librealsense2/rs.hpp>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

struct Options {
    std::string serial;
    int lwir_index = -1;
    int lwir_width = 160;
    int lwir_height = 120;
    int lwir_fps = 25;
    int depth_width = 640;
    int depth_height = 480;
    int depth_fps = 30;
    int color_width = 640;
    int color_height = 480;
    int color_fps = 30;
    double sync_max_delta_ms = 40.0;
    std::size_t sync_buffer_size = 8;
    bool white_hot = false;
    bool show_windows = true;
    std::string d435_calib = "d435if_calib.yaml";
    std::string lepton_calib = "lepton35_calib(1).yaml";
    std::string stereo_extrinsics = "calib_extrinsics(2).yaml";
    bool align_depth_to_lwir = true;
    bool align_rgb_to_lwir = true;
    int align_skip_frames = 1;
};

struct VideoDeviceInfo {
    int index = -1;
    std::string name;
};

struct LwirPacket {
    TimePoint recv_ts;
    cv::Mat frame16;
};

struct RsPacket {
    TimePoint recv_ts;
    cv::Mat color_bgr;
    cv::Mat depth_u16;
};

struct SyncedPair {
    LwirPacket lwir;
    RsPacket rs;
    double delta_ms = 0.0;
};

struct CameraCalib {
    cv::Matx33d K = cv::Matx33d::eye();
    cv::Mat dist = cv::Mat::zeros(1, 5, CV_64F);
    cv::Size size;
};

struct UnifiedCalibration {
    CameraCalib depth;
    CameraCalib rgb;
    CameraCalib ir;
    cv::Matx33d R_depth_to_rgb = cv::Matx33d::eye();
    cv::Vec3d t_depth_to_rgb = {0.0, 0.0, 0.0};
    cv::Matx33d R_rgb_to_ir = cv::Matx33d::eye();
    cv::Vec3d t_rgb_to_ir = {0.0, 0.0, 0.0};
};

struct ProcessedFrames {
    cv::Mat lwir_show;
    cv::Mat depth_colormap;
    cv::Mat color_show;
    cv::Mat depth_aligned_m;
    cv::Mat depth_aligned_colormap;
    cv::Mat rgb_aligned;
};

static cv::Mat normalize_lwir(const cv::Mat& frame16, bool white_hot);
static cv::Mat depth_to_colormap(const cv::Mat& depth_u16);

static int64_t to_ns(const TimePoint& tp) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
}

static double delta_ms(const TimePoint& a, const TimePoint& b) {
    const auto delta = a > b ? a - b : b - a;
    return std::chrono::duration<double, std::milli>(delta).count();
}

static std::vector<VideoDeviceInfo> list_video_devices() {
    std::vector<VideoDeviceInfo> devices;
    const fs::path sysfs_root("/sys/class/video4linux");
    if (!fs::exists(sysfs_root)) {
        return devices;
    }

    for (const auto& entry : fs::directory_iterator(sysfs_root)) {
        const auto name = entry.path().filename().string();
        if (name.rfind("video", 0) != 0) {
            continue;
        }
        try {
            const int index = std::stoi(name.substr(5));
            std::ifstream in(entry.path() / "name");
            std::string device_name;
            std::getline(in, device_name);
            devices.push_back({index, device_name});
        } catch (...) {
            continue;
        }
    }

    std::sort(devices.begin(), devices.end(), [](const auto& a, const auto& b) {
        return a.index < b.index;
    });
    return devices;
}

static std::optional<int> auto_find_lwir_index() {
    const auto devices = list_video_devices();
    for (const auto& device : devices) {
        std::string lower = device.name;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (lower.find("purethermal") != std::string::npos ||
            lower.find("lepton") != std::string::npos ||
            lower.find("flir") != std::string::npos) {
            return device.index;
        }
    }
    return std::nullopt;
}

static fs::path executable_dir() {
    return fs::canonical("/proc/self/exe").parent_path();
}

static fs::path source_dir() {
    return fs::path(__FILE__).parent_path();
}

static fs::path expand_user_path(const std::string& path) {
    if (!path.empty() && path[0] == '~') {
        const char* home = std::getenv("HOME");
        if (home != nullptr) {
            if (path.size() == 1) {
                return fs::path(home);
            }
            if (path[1] == '/') {
                return fs::path(home) / path.substr(2);
            }
        }
    }
    return fs::path(path);
}

static fs::path resolve_input_path(const std::string& path) {
    const fs::path candidate = expand_user_path(path);
    if (fs::exists(candidate)) {
        return fs::canonical(candidate);
    }
    const fs::path exec_relative = executable_dir() / candidate;
    if (fs::exists(exec_relative)) {
        return fs::canonical(exec_relative);
    }
    const fs::path src_relative = source_dir() / candidate;
    if (fs::exists(src_relative)) {
        return fs::canonical(src_relative);
    }
    throw std::runtime_error("file not found: " + path);
}

static cv::Vec<double, 5> node_to_dist(const YAML::Node& node) {
    cv::Vec<double, 5> coeffs{0.0, 0.0, 0.0, 0.0, 0.0};
    if (!node || !node.IsSequence()) {
        return coeffs;
    }
    const std::size_t count = std::min<std::size_t>(5, node.size());
    for (std::size_t i = 0; i < count; ++i) {
        coeffs[static_cast<int>(i)] = node[i].as<double>();
    }
    return coeffs;
}

static cv::Mat dist_to_mat(const cv::Vec<double, 5>& dist) {
    return cv::Mat(dist).reshape(1, 1).clone();
}

static cv::Matx33d node_to_mat3(const YAML::Node& node) {
    cv::Matx33d out = cv::Matx33d::eye();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out(r, c) = node[r][c].as<double>();
        }
    }
    return out;
}

static cv::Vec3d node_to_vec3(const YAML::Node& node) {
    return {node[0].as<double>(), node[1].as<double>(), node[2].as<double>()};
}

static CameraCalib load_camera_node(const YAML::Node& node) {
    CameraCalib calib;
    calib.size = cv::Size(node["width"].as<int>(), node["height"].as<int>());
    calib.K = cv::Matx33d(
        node["fx"].as<double>(), 0.0, node["cx"].as<double>(),
        0.0, node["fy"].as<double>(), node["cy"].as<double>(),
        0.0, 0.0, 1.0
    );
    calib.dist = dist_to_mat(node_to_dist(node["distortion_coefficients"]));
    return calib;
}

static UnifiedCalibration load_unified_calibration(const Options& options) {
    const auto d435_yaml = YAML::LoadFile(resolve_input_path(options.d435_calib).string());
    const auto lepton_yaml = YAML::LoadFile(resolve_input_path(options.lepton_calib).string());
    const auto stereo_yaml = YAML::LoadFile(resolve_input_path(options.stereo_extrinsics).string());

    UnifiedCalibration calib;
    calib.depth = load_camera_node(d435_yaml["depth_camera"]);
    calib.rgb = load_camera_node(d435_yaml["color_camera"]);
    calib.ir = load_camera_node(lepton_yaml["thermal_camera"]);
    calib.R_depth_to_rgb = node_to_mat3(d435_yaml["extrinsics"]["depth_to_color"]["rotation"]);
    calib.t_depth_to_rgb = node_to_vec3(d435_yaml["extrinsics"]["depth_to_color"]["translation"]);
    calib.R_rgb_to_ir = node_to_mat3(stereo_yaml["extrinsics_rgb_to_ir"]["R"]);
    calib.t_rgb_to_ir = node_to_vec3(stereo_yaml["extrinsics_rgb_to_ir"]["t"]);
    return calib;
}

class SoftSynchronizer {
public:
    SoftSynchronizer(std::size_t max_buffer_size, double max_delta_ms)
        : max_buffer_size_(std::max<std::size_t>(1, max_buffer_size)),
          max_delta_ns_(static_cast<int64_t>(max_delta_ms * 1e6)) {}

    void push_lwir(LwirPacket packet) {
        std::scoped_lock lock(mutex_);
        push_bounded(lwir_buffer_, std::move(packet));
        drop_stale_locked();
    }

    void push_rs(RsPacket packet) {
        std::scoped_lock lock(mutex_);
        push_bounded(rs_buffer_, std::move(packet));
        drop_stale_locked();
    }

    std::optional<SyncedPair> pop_best_pair() {
        std::scoped_lock lock(mutex_);
        if (lwir_buffer_.empty() || rs_buffer_.empty()) {
            return std::nullopt;
        }

        std::optional<std::pair<std::size_t, std::size_t>> best_indices;
        int64_t best_delta = 0;

        for (std::size_t i = 0; i < lwir_buffer_.size(); ++i) {
            for (std::size_t j = 0; j < rs_buffer_.size(); ++j) {
                const int64_t delta = std::llabs(to_ns(lwir_buffer_[i].recv_ts) - to_ns(rs_buffer_[j].recv_ts));
                if (delta > max_delta_ns_) {
                    continue;
                }
                if (!best_indices || delta < best_delta) {
                    best_indices = std::make_pair(i, j);
                    best_delta = delta;
                }
            }
        }

        if (!best_indices) {
            drop_stale_locked();
            return std::nullopt;
        }

        auto lwir = remove_at(lwir_buffer_, best_indices->first);
        auto rs = remove_at(rs_buffer_, best_indices->second);
        return SyncedPair{std::move(lwir), std::move(rs), static_cast<double>(best_delta) / 1e6};
    }

private:
    template <typename T>
    void push_bounded(std::deque<T>& buffer, T value) {
        if (buffer.size() >= max_buffer_size_) {
            buffer.pop_front();
        }
        buffer.push_back(std::move(value));
    }

    template <typename T>
    T remove_at(std::deque<T>& buffer, std::size_t index) {
        T value = std::move(buffer[index]);
        buffer.erase(buffer.begin() + static_cast<std::ptrdiff_t>(index));
        return value;
    }

    void drop_stale_locked() {
        if (lwir_buffer_.empty() || rs_buffer_.empty()) {
            return;
        }
        const int64_t newest_lwir = to_ns(lwir_buffer_.back().recv_ts);
        const int64_t newest_rs = to_ns(rs_buffer_.back().recv_ts);

        while (!lwir_buffer_.empty() && to_ns(lwir_buffer_.front().recv_ts) < newest_rs - max_delta_ns_) {
            lwir_buffer_.pop_front();
        }
        while (!rs_buffer_.empty() && to_ns(rs_buffer_.front().recv_ts) < newest_lwir - max_delta_ns_) {
            rs_buffer_.pop_front();
        }
    }

    std::mutex mutex_;
    std::deque<LwirPacket> lwir_buffer_;
    std::deque<RsPacket> rs_buffer_;
    std::size_t max_buffer_size_;
    int64_t max_delta_ns_;
};

class GeometryProcessor {
public:
    explicit GeometryProcessor(UnifiedCalibration calib)
        : calib_(std::move(calib)) {
        cv::Rodrigues(cv::Mat::eye(3, 3, CV_64F), zero_rvec_);
        zero_tvec_ = cv::Mat::zeros(3, 1, CV_64F);
    }

    ProcessedFrames process(
        const SyncedPair& pair,
        bool white_hot,
        bool enable_align_depth_to_lwir,
        bool enable_align_rgb_to_lwir,
        int frame_count,
        int align_skip_frames
    ) {
        ProcessedFrames out;
        out.lwir_show = normalize_lwir(pair.lwir.frame16, white_hot);
        cv::resize(out.lwir_show, out.lwir_show, cv::Size(640, 480), 0.0, 0.0, cv::INTER_LINEAR);
        out.color_show = pair.rs.color_bgr;
        out.depth_colormap = depth_to_colormap(pair.rs.depth_u16);

        if (enable_align_depth_to_lwir) {
            if (cached_depth_aligned_m_.empty() || frame_count % std::max(1, align_skip_frames) == 0) {
                cached_depth_aligned_m_ = align_depth_to_lwir_fast(pair.rs.depth_u16);
                cached_depth_colormap_ = depth_float_to_colormap(cached_depth_aligned_m_);
            }
            out.depth_aligned_m = cached_depth_aligned_m_;
            out.depth_aligned_colormap = cached_depth_colormap_;
        }

        if (enable_align_rgb_to_lwir && !cached_depth_aligned_m_.empty()) {
            if (cached_rgb_aligned_.empty() || frame_count % std::max(1, align_skip_frames) == 0) {
                cached_rgb_aligned_ = align_rgb_to_lwir_impl(pair.rs.color_bgr, cached_depth_aligned_m_);
            }
            out.rgb_aligned = cached_rgb_aligned_;
        }
        return out;
    }

    const UnifiedCalibration& calib() const {
        return calib_;
    }

private:
    cv::Mat align_depth_to_lwir_fast(const cv::Mat& depth_raw) const {
        const int ir_w = calib_.ir.size.width;
        const int ir_h = calib_.ir.size.height;
        const int stride = 2;

        cv::Mat depth_m;
        depth_raw.convertTo(depth_m, CV_32F, 0.001);

        std::vector<cv::Point2f> pixels;
        std::vector<float> z_values;
        pixels.reserve((depth_raw.rows / stride) * (depth_raw.cols / stride));
        z_values.reserve((depth_raw.rows / stride) * (depth_raw.cols / stride));

        for (int y = 0; y < depth_raw.rows; y += stride) {
            const float* row = depth_m.ptr<float>(y);
            for (int x = 0; x < depth_raw.cols; x += stride) {
                const float z = row[x];
                if (z > 0.1f) {
                    pixels.emplace_back(static_cast<float>(x), static_cast<float>(y));
                    z_values.push_back(z);
                }
            }
        }
        if (pixels.empty()) {
            return cv::Mat::zeros(ir_h, ir_w, CV_32F);
        }

        cv::Mat pts_pixel(static_cast<int>(pixels.size()), 1, CV_32FC2, pixels.data());
        cv::Mat pts_undist;
        cv::undistortPoints(pts_pixel, pts_undist, cv::Mat(calib_.depth.K), calib_.depth.dist);
        const auto* undist = pts_undist.ptr<cv::Vec2f>();

        std::vector<cv::Point3f> pts_ir;
        pts_ir.reserve(pixels.size());
        for (std::size_t i = 0; i < pixels.size(); ++i) {
            const double z = z_values[i];
            const cv::Vec3d p_depth(undist[i][0] * z, undist[i][1] * z, z);
            const cv::Vec3d p_rgb = calib_.R_depth_to_rgb * p_depth + calib_.t_depth_to_rgb;
            const cv::Vec3d p_ir = calib_.R_rgb_to_ir * p_rgb + calib_.t_rgb_to_ir;
            if (p_ir[2] > 0.1) {
                pts_ir.emplace_back(
                    static_cast<float>(p_ir[0]),
                    static_cast<float>(p_ir[1]),
                    static_cast<float>(p_ir[2])
                );
            } else {
                pts_ir.emplace_back(0.0f, 0.0f, -1.0f);
            }
        }

        std::vector<cv::Point2f> projected;
        cv::projectPoints(pts_ir, zero_rvec_, zero_tvec_, cv::Mat(calib_.ir.K), calib_.ir.dist, projected);

        cv::Mat depth_ir(ir_h, ir_w, CV_32F, cv::Scalar(0.0f));
        cv::Mat zbuffer(ir_h, ir_w, CV_32F, cv::Scalar(std::numeric_limits<float>::infinity()));
        for (std::size_t i = 0; i < projected.size(); ++i) {
            const auto& p = projected[i];
            const auto& xyz = pts_ir[i];
            if (xyz.z <= 0.1f) {
                continue;
            }
            const int u = static_cast<int>(std::lround(p.x));
            const int v = static_cast<int>(std::lround(p.y));
            if (u < 0 || u >= ir_w || v < 0 || v >= ir_h) {
                continue;
            }
            float& best = zbuffer.at<float>(v, u);
            if (xyz.z < best) {
                best = xyz.z;
                depth_ir.at<float>(v, u) = xyz.z;
            }
        }
        cv::medianBlur(depth_ir, depth_ir, 3);
        return depth_ir;
    }

    cv::Mat align_rgb_to_lwir_impl(const cv::Mat& rgb, const cv::Mat& depth_aligned_m) const {
        cv::Mat depth_resized;
        cv::resize(depth_aligned_m, depth_resized, cv::Size(640, 480), 0.0, 0.0, cv::INTER_NEAREST);

        cv::Mat rgb_out(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Matx33d K_ir = calib_.ir.K;
        K_ir(0, 0) *= 640.0 / static_cast<double>(calib_.ir.size.width);
        K_ir(1, 1) *= 480.0 / static_cast<double>(calib_.ir.size.height);
        K_ir(0, 2) *= 640.0 / static_cast<double>(calib_.ir.size.width);
        K_ir(1, 2) *= 480.0 / static_cast<double>(calib_.ir.size.height);

        const cv::Matx33d R_i2r = calib_.R_rgb_to_ir.t();
        const cv::Vec3d t_i2r = -(R_i2r * calib_.t_rgb_to_ir);

        for (int v = 0; v < depth_resized.rows; v += 2) {
            const float* row = depth_resized.ptr<float>(v);
            for (int u = 0; u < depth_resized.cols; u += 2) {
                const double z = row[u];
                if (z <= 0.1) {
                    continue;
                }
                const double x_ir = (u - K_ir(0, 2)) * z / K_ir(0, 0);
                const double y_ir = (v - K_ir(1, 2)) * z / K_ir(1, 1);
                const cv::Vec3d p_ir(x_ir, y_ir, z);
                const cv::Vec3d p_rgb = R_i2r * p_ir + t_i2r;
                if (p_rgb[2] <= 0.1) {
                    continue;
                }

                const int u_rgb = static_cast<int>(std::lround(p_rgb[0] * calib_.rgb.K(0, 0) / p_rgb[2] + calib_.rgb.K(0, 2)));
                const int v_rgb = static_cast<int>(std::lround(p_rgb[1] * calib_.rgb.K(1, 1) / p_rgb[2] + calib_.rgb.K(1, 2)));
                if (u_rgb < 0 || u_rgb >= rgb.cols || v_rgb < 0 || v_rgb >= rgb.rows) {
                    continue;
                }
                rgb_out.at<cv::Vec3b>(v, u) = rgb.at<cv::Vec3b>(v_rgb, u_rgb);
            }
        }
        return rgb_out;
    }

    static cv::Mat depth_float_to_colormap(const cv::Mat& depth_m) {
        cv::Mat depth_8u;
        cv::convertScaleAbs(depth_m, depth_8u, 25.5);
        cv::Mat color;
        cv::applyColorMap(depth_8u, color, cv::COLORMAP_JET);
        return color;
    }

    UnifiedCalibration calib_;
    cv::Mat zero_rvec_;
    cv::Mat zero_tvec_;
    cv::Mat cached_depth_aligned_m_;
    cv::Mat cached_depth_colormap_;
    cv::Mat cached_rgb_aligned_;
};

static cv::Mat normalize_lwir(const cv::Mat& frame16, bool white_hot) {
    cv::Mat cropped = frame16;
    if (cropped.rows > 120) {
        cropped = cropped.rowRange(0, 120);
    }

    double min_val = 0.0;
    double max_val = 0.0;
    cv::minMaxLoc(cropped, &min_val, &max_val);
    if (max_val - min_val < 1.0) {
        max_val = min_val + 1.0;
    }

    cv::Mat normalized;
    cropped.convertTo(normalized, CV_8U, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));
    if (!white_hot) {
        normalized = 255 - normalized;
    }
    cv::flip(normalized, normalized, -1);
    cv::medianBlur(normalized, normalized, 3);
    return normalized;
}

static cv::Mat depth_to_colormap(const cv::Mat& depth_u16) {
    cv::Mat depth_8u;
    cv::convertScaleAbs(depth_u16, depth_8u, 0.03);
    cv::Mat colored;
    cv::applyColorMap(depth_8u, colored, cv::COLORMAP_JET);
    return colored;
}

static void lwir_capture_loop(const Options& options, SoftSynchronizer& sync, std::atomic<bool>& running) {
    int lwir_index = options.lwir_index;
    if (lwir_index < 0) {
        const auto detected = auto_find_lwir_index();
        if (!detected) {
            std::cerr << "[错误] 未找到 PureThermal/Lepton 设备\n";
            running.store(false);
            return;
        }
        lwir_index = *detected;
        std::cout << "[信息] 自动检测到 LWIR 设备: /dev/video" << lwir_index << "\n";
    }

    cv::VideoCapture cap(lwir_index, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "[错误] 无法打开 LWIR 设备 /dev/video" << lwir_index << "\n";
        running.store(false);
        return;
    }

    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap.set(cv::CAP_PROP_CONVERT_RGB, 0);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', '1', '6', ' '));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, options.lwir_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, options.lwir_height);
    cap.set(cv::CAP_PROP_FPS, options.lwir_fps);

    for (int i = 0; i < 5; ++i) {
        cv::Mat warmup;
        cap.read(warmup);
    }

    while (running.load()) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        if (frame.type() != CV_16UC1) {
            if (frame.channels() == 2) {
                frame = frame.reshape(1);
            } else {
                std::cerr << "[警告] LWIR 帧格式异常, type=" << frame.type() << "\n";
                continue;
            }
        }
        sync.push_lwir(LwirPacket{Clock::now(), frame.clone()});
    }
}

static void rs_capture_loop(const Options& options, SoftSynchronizer& sync, std::atomic<bool>& running) {
    rs2::pipeline pipe;
    rs2::config config;
    if (!options.serial.empty()) {
        config.enable_device(options.serial);
    }
    config.enable_stream(RS2_STREAM_DEPTH, options.depth_width, options.depth_height, RS2_FORMAT_Z16, options.depth_fps);
    config.enable_stream(RS2_STREAM_COLOR, options.color_width, options.color_height, RS2_FORMAT_BGR8, options.color_fps);

    try {
        pipe.start(config);
    } catch (const rs2::error& e) {
        std::cerr << "[错误] RealSense 启动失败: " << e.what() << "\n";
        running.store(false);
        return;
    }

    while (running.load()) {
        try {
            rs2::frameset frames;
            if (!pipe.poll_for_frames(&frames)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            const TimePoint recv_ts = Clock::now();
            auto depth = frames.get_depth_frame();
            auto color = frames.get_color_frame();
            if (!depth || !color) {
                continue;
            }

            const cv::Mat depth_view(
                cv::Size(depth.get_width(), depth.get_height()),
                CV_16UC1,
                const_cast<void*>(depth.get_data()),
                cv::Mat::AUTO_STEP
            );
            const cv::Mat color_view(
                cv::Size(color.get_width(), color.get_height()),
                CV_8UC3,
                const_cast<void*>(color.get_data()),
                cv::Mat::AUTO_STEP
            );

            sync.push_rs(RsPacket{recv_ts, color_view.clone(), depth_view.clone()});
        } catch (const rs2::error& e) {
            std::cerr << "[警告] RealSense 取帧失败: " << e.what() << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    try {
        pipe.stop();
    } catch (...) {
    }
}

static void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "  --serial <id>               RealSense serial\n"
        << "  --lwir-index <n>            Lepton /dev/video index, -1 auto-detect\n"
        << "  --sync-max-delta-ms <ms>    Max timestamp delta for pairing\n"
        << "  --sync-buffer-size <n>      Per-stream ring buffer size\n"
        << "  --d435-calib <path>         D435 calibration yaml\n"
        << "  --lepton-calib <path>       Lepton calibration yaml\n"
        << "  --stereo-extrinsics <path>  Stereo extrinsics yaml\n"
        << "  --align-skip-frames <n>     Recompute geometry every n synced frames\n"
        << "  --white-hot                 Show thermal targets as white\n"
        << "  --no-display                Disable OpenCV windows\n";
}

static bool parse_args(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--serial") {
            options.serial = require_value(arg);
        } else if (arg == "--lwir-index") {
            options.lwir_index = std::stoi(require_value(arg));
        } else if (arg == "--sync-max-delta-ms") {
            options.sync_max_delta_ms = std::stod(require_value(arg));
        } else if (arg == "--sync-buffer-size") {
            options.sync_buffer_size = static_cast<std::size_t>(std::stoul(require_value(arg)));
        } else if (arg == "--d435-calib") {
            options.d435_calib = require_value(arg);
        } else if (arg == "--lepton-calib") {
            options.lepton_calib = require_value(arg);
        } else if (arg == "--stereo-extrinsics") {
            options.stereo_extrinsics = require_value(arg);
        } else if (arg == "--align-skip-frames") {
            options.align_skip_frames = std::stoi(require_value(arg));
        } else if (arg == "--no-align-depth-to-lwir") {
            options.align_depth_to_lwir = false;
        } else if (arg == "--no-align-rgb-to-lwir") {
            options.align_rgb_to_lwir = false;
        } else if (arg == "--white-hot") {
            options.white_hot = true;
        } else if (arg == "--no-display") {
            options.show_windows = false;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return true;
}

#ifndef MULTICAM_SYNC_NO_MAIN
int main(int argc, char** argv) {
    Options options;
    try {
        if (!parse_args(argc, argv, options)) {
            return 0;
        }
    } catch (const std::exception& e) {
        std::cerr << "[错误] 参数解析失败: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    const auto devices = list_video_devices();
    if (!devices.empty()) {
        std::cout << "[信息] 检测到视频设备:\n";
        for (const auto& device : devices) {
            std::cout << "  /dev/video" << device.index << " : " << device.name << "\n";
        }
    }

    SoftSynchronizer synchronizer(options.sync_buffer_size, options.sync_max_delta_ms);
    GeometryProcessor processor(load_unified_calibration(options));
    std::atomic<bool> running{true};

    std::thread lwir_thread(lwir_capture_loop, std::cref(options), std::ref(synchronizer), std::ref(running));
    std::thread rs_thread(rs_capture_loop, std::cref(options), std::ref(synchronizer), std::ref(running));

    double fps = 0.0;
    int synced_count = 0;
    int frame_count = 0;
    auto fps_start = Clock::now();

    while (running.load()) {
        auto pair = synchronizer.pop_best_pair();
        if (!pair) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        ++synced_count;
        ++frame_count;
        const auto now = Clock::now();
        const double elapsed = std::chrono::duration<double>(now - fps_start).count();
        if (elapsed >= 1.0) {
            fps = synced_count / elapsed;
            synced_count = 0;
            fps_start = now;
        }

        if (!options.show_windows) {
            std::cout << "[sync] delta_ms=" << std::fixed << std::setprecision(3) << pair->delta_ms << "\n";
            continue;
        }

        ProcessedFrames processed = processor.process(
            *pair,
            options.white_hot,
            options.align_depth_to_lwir,
            options.align_rgb_to_lwir,
            frame_count,
            options.align_skip_frames
        );

        std::ostringstream sync_text;
        sync_text << "sync=" << std::fixed << std::setprecision(2) << pair->delta_ms << "ms fps=" << fps;
        cv::putText(processed.lwir_show, sync_text.str(), {10, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {255}, 2);
        cv::putText(processed.color_show, sync_text.str(), {10, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2);
        cv::putText(processed.depth_colormap, sync_text.str(), {10, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2);

        cv::imshow("LWIR", processed.lwir_show);
        cv::imshow("RealSense RGB", processed.color_show);
        cv::imshow("RealSense Depth", processed.depth_colormap);
        if (!processed.rgb_aligned.empty()) {
            cv::imshow("Aligned RGB to IR", processed.rgb_aligned);
        }
        if (!processed.depth_aligned_colormap.empty()) {
            cv::imshow("Aligned Depth to IR", processed.depth_aligned_colormap);
        }

        const int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {
            running.store(false);
            break;
        }
    }

    running.store(false);
    if (lwir_thread.joinable()) {
        lwir_thread.join();
    }
    if (rs_thread.joinable()) {
        rs_thread.join();
    }
    cv::destroyAllWindows();
    return 0;
}
#endif
