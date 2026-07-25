#define MULTICAM_SYNC_NO_MAIN
#include "multicam_sync.cpp"

#include <cstring>
#include <condition_variable>
#include <deque>

#include <Python.h>
#include <frameobject.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

struct PythonFramePacket {
    bool synced = true;
    cv::Mat lwir;
    cv::Mat lwir_raw16;
    cv::Mat color;
    cv::Mat depth_raw;
    cv::Mat depth_colormap;
    cv::Mat rgb_aligned;
    cv::Mat depth_aligned_m;
    cv::Mat depth_aligned_colormap;
    double sync_delta_ms = 0.0;
    int64_t lwir_ts_ns = 0;
    int64_t rs_ts_ns = 0;
};

class PyMultiCamSync {
public:
    PyMultiCamSync(
        const std::string& serial = "",
        int lwir_index = -1,
        int lwir_width = 160,
        int lwir_height = 120,
        int lwir_fps = 25,
        int depth_width = 640,
        int depth_height = 480,
        int depth_fps = 30,
        int color_width = 640,
        int color_height = 480,
        int color_fps = 30,
        double sync_max_delta_ms = 40.0,
        std::size_t sync_buffer_size = 8,
        bool white_hot = false,
        const std::string& d435_calib = "d435if_calib.yaml",
        const std::string& lepton_calib = "lepton35_calib(1).yaml",
        const std::string& stereo_extrinsics = "calib_extrinsics(2).yaml",
        bool align_depth_to_lwir = true,
        bool align_rgb_to_lwir = true,
        int align_skip_frames = 1
    ) {
        options_.serial = serial;
        options_.lwir_index = lwir_index;
        options_.lwir_width = lwir_width;
        options_.lwir_height = lwir_height;
        options_.lwir_fps = lwir_fps;
        options_.depth_width = depth_width;
        options_.depth_height = depth_height;
        options_.depth_fps = depth_fps;
        options_.color_width = color_width;
        options_.color_height = color_height;
        options_.color_fps = color_fps;
        options_.sync_max_delta_ms = sync_max_delta_ms;
        options_.sync_buffer_size = sync_buffer_size;
        options_.white_hot = white_hot;
        options_.show_windows = false;
        options_.d435_calib = d435_calib;
        options_.lepton_calib = lepton_calib;
        options_.stereo_extrinsics = stereo_extrinsics;
        options_.align_depth_to_lwir = align_depth_to_lwir;
        options_.align_rgb_to_lwir = align_rgb_to_lwir;
        options_.align_skip_frames = align_skip_frames;
    }

    ~PyMultiCamSync() {
        stop();
    }

    void start() {
        if (running_.load()) {
            return;
        }
        synchronizer_ = std::make_unique<SoftSynchronizer>(options_.sync_buffer_size, options_.sync_max_delta_ms);
        calib_ = load_unified_calibration(options_);
        processor_ = std::make_unique<GeometryProcessor>(calib_);
        running_.store(true);
        frame_count_ = 0;
        lwir_frames_.store(0);
        rs_frames_.store(0);
        matched_frames_.store(0);
        sync_timeouts_.store(0);

        lwir_thread_ = std::thread(&PyMultiCamSync::lwir_capture_loop_member, this);
        rs_thread_ = std::thread(&PyMultiCamSync::rs_capture_loop_member, this);
        process_thread_ = std::thread(&PyMultiCamSync::process_loop, this);
    }

    void stop() {
        if (!running_.load() && !lwir_thread_.joinable() && !rs_thread_.joinable() && !process_thread_.joinable()) {
            return;
        }

        running_.store(false);
        cv_.notify_all();

        if (lwir_thread_.joinable()) {
            lwir_thread_.join();
        }
        if (rs_thread_.joinable()) {
            rs_thread_.join();
        }
        if (process_thread_.joinable()) {
            process_thread_.join();
        }

        {
            std::scoped_lock lock(mutex_);
            queue_.clear();
        }
        synchronizer_.reset();
        processor_.reset();
    }

    py::object read(int timeout_ms = 1000) {
        PythonFramePacket packet;
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> lock(mutex_);
            const bool ready = cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&]() {
                return !queue_.empty() || !running_.load();
            });
            if (!ready || queue_.empty()) {
                sync_timeouts_.fetch_add(1);
                packet = latest_preview_packet_locked();
                if (packet.color.empty() && packet.lwir.empty()) {
                    return py::none();
                }
            } else {
                packet = std::move(queue_.front());
                queue_.pop_front();
            }
        }
        py::dict out;
        out["synced"] = packet.synced;
        out["lwir"] = mat_to_numpy(packet.lwir);
        out["lwir_raw16"] = packet.lwir_raw16.empty() ? py::object(py::none()) : py::object(mat_to_numpy(packet.lwir_raw16));
        out["color"] = mat_to_numpy(packet.color);
        out["color_uncropped"] = mat_to_numpy(packet.color);
        out["depth_raw"] = mat_to_numpy(packet.depth_raw);
        out["depth_colormap"] = packet.depth_colormap.empty() ? py::object(py::none()) : py::object(mat_to_numpy(packet.depth_colormap));
        out["rgb_aligned"] = packet.rgb_aligned.empty() ? py::object(py::none()) : py::object(mat_to_numpy(packet.rgb_aligned));
        out["depth_aligned_m"] = packet.depth_aligned_m.empty() ? py::object(py::none()) : py::object(mat_to_numpy(packet.depth_aligned_m));
        out["depth_aligned_colormap"] = packet.depth_aligned_colormap.empty() ? py::object(py::none()) : py::object(mat_to_numpy(packet.depth_aligned_colormap));
        out["lwir_ts_ns"] = packet.lwir_ts_ns;
        out["rs_ts_ns"] = packet.rs_ts_ns;
        out["sync_delta_ms"] = packet.sync_delta_ms;
        out["depth_scale"] = 0.001;
        out["stats"] = py::dict(
            "lwir_frames"_a=lwir_frames_.load(),
            "rs_frames"_a=rs_frames_.load(),
            "matched_frames"_a=matched_frames_.load(),
            "sync_timeouts"_a=sync_timeouts_.load()
        );
        out["depth_intrin"] = py::dict(
            "fx"_a=calib_.depth.K(0, 0),
            "fy"_a=calib_.depth.K(1, 1),
            "cx"_a=calib_.depth.K(0, 2),
            "cy"_a=calib_.depth.K(1, 2)
        );
        return std::move(out);
    }

private:
    void lwir_capture_loop_member() {
        int lwir_index = options_.lwir_index;
        if (lwir_index < 0) {
            const auto detected = auto_find_lwir_index();
            if (!detected) {
                running_.store(false);
                cv_.notify_all();
                return;
            }
            lwir_index = *detected;
        }

        cv::VideoCapture cap(lwir_index, cv::CAP_V4L2);
        if (!cap.isOpened()) {
            running_.store(false);
            cv_.notify_all();
            return;
        }

        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        cap.set(cv::CAP_PROP_CONVERT_RGB, 0);
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', '1', '6', ' '));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, options_.lwir_width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, options_.lwir_height);
        cap.set(cv::CAP_PROP_FPS, options_.lwir_fps);

        for (int i = 0; i < 5; ++i) {
            cv::Mat warmup;
            cap.read(warmup);
        }

        while (running_.load()) {
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            if (frame.type() != CV_16UC1) {
                if (frame.channels() == 2) {
                    frame = frame.reshape(1);
                } else {
                    continue;
                }
            }
            const auto ts = Clock::now();
            synchronizer_->push_lwir(LwirPacket{ts, frame.clone()});
            lwir_frames_.fetch_add(1);

            std::scoped_lock lock(mutex_);
            latest_lwir_raw16_ = frame.clone();
            latest_lwir_ts_ns_ = to_ns(ts);
        }
    }

    void rs_capture_loop_member() {
        rs2::pipeline pipe;
        rs2::config config;
        if (!options_.serial.empty()) {
            config.enable_device(options_.serial);
        }
        config.enable_stream(RS2_STREAM_DEPTH, options_.depth_width, options_.depth_height, RS2_FORMAT_Z16, options_.depth_fps);
        config.enable_stream(RS2_STREAM_COLOR, options_.color_width, options_.color_height, RS2_FORMAT_BGR8, options_.color_fps);

        try {
            pipe.start(config);
        } catch (...) {
            running_.store(false);
            cv_.notify_all();
            return;
        }

        while (running_.load()) {
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

                cv::Mat depth_clone = depth_view.clone();
                cv::Mat color_clone = color_view.clone();
                synchronizer_->push_rs(RsPacket{recv_ts, color_clone, depth_clone});
                rs_frames_.fetch_add(1);

                std::scoped_lock lock(mutex_);
                latest_color_ = color_clone;
                latest_depth_raw_ = depth_clone;
                latest_rs_ts_ns_ = to_ns(recv_ts);
            } catch (...) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }

        try {
            pipe.stop();
        } catch (...) {
        }
    }

    PythonFramePacket latest_preview_packet_locked() const {
        PythonFramePacket packet;
        packet.synced = false;
        packet.lwir_raw16 = latest_lwir_raw16_.clone();
        packet.color = latest_color_.clone();
        packet.depth_raw = latest_depth_raw_.clone();
        packet.lwir_ts_ns = latest_lwir_ts_ns_;
        packet.rs_ts_ns = latest_rs_ts_ns_;
        if (!latest_lwir_raw16_.empty()) {
            packet.lwir = normalize_lwir(latest_lwir_raw16_, options_.white_hot);
            cv::resize(packet.lwir, packet.lwir, cv::Size(640, 480), 0.0, 0.0, cv::INTER_LINEAR);
        }
        return packet;
    }

    void process_loop() {
        while (running_.load()) {
            if (!synchronizer_ || !processor_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            auto pair = synchronizer_->pop_best_pair();
            if (!pair) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            ++frame_count_;
            matched_frames_.fetch_add(1);
            ProcessedFrames processed = processor_->process(
                *pair,
                options_.white_hot,
                options_.align_depth_to_lwir,
                options_.align_rgb_to_lwir,
                frame_count_,
                options_.align_skip_frames
            );

            PythonFramePacket packet;
            packet.lwir = processed.lwir_show.clone();
            packet.lwir_raw16 = pair->lwir.frame16.clone();
            packet.color = processed.color_show.clone();
            packet.depth_raw = pair->rs.depth_u16.clone();
            packet.depth_colormap = processed.depth_colormap.clone();
            packet.rgb_aligned = processed.rgb_aligned.clone();
            packet.depth_aligned_m = processed.depth_aligned_m.clone();
            packet.depth_aligned_colormap = processed.depth_aligned_colormap.clone();
            packet.sync_delta_ms = pair->delta_ms;
            packet.lwir_ts_ns = to_ns(pair->lwir.recv_ts);
            packet.rs_ts_ns = to_ns(pair->rs.recv_ts);

            {
                std::scoped_lock lock(mutex_);
                if (queue_.size() >= 3) {
                    queue_.pop_front();
                }
                queue_.push_back(std::move(packet));
            }
            cv_.notify_one();
        }
    }

    static py::array mat_to_numpy(const cv::Mat& mat) {
        if (mat.empty()) {
            return py::array();
        }

        if (mat.type() == CV_8UC1) {
            py::array_t<uint8_t> arr({mat.rows, mat.cols});
            std::memcpy(arr.mutable_data(), mat.data, static_cast<std::size_t>(mat.rows * mat.cols));
            return arr;
        }
        if (mat.type() == CV_8UC3) {
            py::array_t<uint8_t> arr({mat.rows, mat.cols, 3});
            std::memcpy(arr.mutable_data(), mat.data, static_cast<std::size_t>(mat.rows * mat.cols * 3));
            return arr;
        }
        if (mat.type() == CV_16UC1) {
            py::array_t<uint16_t> arr({mat.rows, mat.cols});
            std::memcpy(arr.mutable_data(), mat.data, static_cast<std::size_t>(mat.rows * mat.cols * sizeof(uint16_t)));
            return arr;
        }
        if (mat.type() == CV_32FC1) {
            py::array_t<float> arr({mat.rows, mat.cols});
            std::memcpy(arr.mutable_data(), mat.data, static_cast<std::size_t>(mat.rows * mat.cols * sizeof(float)));
            return arr;
        }
        throw std::runtime_error("unsupported cv::Mat type for numpy export");
    }

    Options options_;
    std::atomic<bool> running_{false};
    std::unique_ptr<SoftSynchronizer> synchronizer_;
    std::unique_ptr<GeometryProcessor> processor_;
    UnifiedCalibration calib_;
    std::thread lwir_thread_;
    std::thread rs_thread_;
    std::thread process_thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<PythonFramePacket> queue_;
    int frame_count_ = 0;
    cv::Mat latest_lwir_raw16_;
    cv::Mat latest_color_;
    cv::Mat latest_depth_raw_;
    int64_t latest_lwir_ts_ns_ = 0;
    int64_t latest_rs_ts_ns_ = 0;
    std::atomic<uint64_t> lwir_frames_{0};
    std::atomic<uint64_t> rs_frames_{0};
    std::atomic<uint64_t> matched_frames_{0};
    std::atomic<uint64_t> sync_timeouts_{0};
};

PYBIND11_MODULE(multicam_sync_py, m) {
    m.doc() = "Low-latency synchronized LWIR + RealSense capture with alignment";

    py::class_<PyMultiCamSync>(m, "MultiCamSync")
        .def(py::init<
             const std::string&, int, int, int, int, int, int, int, int, int, int,
             double, std::size_t, bool, const std::string&, const std::string&, const std::string&,
             bool, bool, int>(),
             py::arg("serial") = "",
             py::arg("lwir_index") = -1,
             py::arg("lwir_width") = 160,
             py::arg("lwir_height") = 120,
             py::arg("lwir_fps") = 25,
             py::arg("depth_width") = 640,
             py::arg("depth_height") = 480,
             py::arg("depth_fps") = 30,
             py::arg("color_width") = 640,
             py::arg("color_height") = 480,
             py::arg("color_fps") = 30,
             py::arg("sync_max_delta_ms") = 40.0,
             py::arg("sync_buffer_size") = 8,
             py::arg("white_hot") = false,
             py::arg("d435_calib") = "d435if_calib.yaml",
             py::arg("lepton_calib") = "lepton35_calib(1).yaml",
             py::arg("stereo_extrinsics") = "calib_extrinsics(2).yaml",
             py::arg("align_depth_to_lwir") = true,
             py::arg("align_rgb_to_lwir") = true,
             py::arg("align_skip_frames") = 1)
        .def("start", &PyMultiCamSync::start)
        .def("stop", &PyMultiCamSync::stop)
        .def("read", &PyMultiCamSync::read, py::arg("timeout_ms") = 1000);
}
