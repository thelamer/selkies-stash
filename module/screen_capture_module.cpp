#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iomanip>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include <X11/Xlib.h>
#include <X11/extensions/XShm.h>
#include <X11/Xutil.h>
#include <jpeglib.h>
#include <netinet/in.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <xxhash.h>

struct CaptureSettings {
    int capture_width;
    int capture_height;
    int capture_x;
    int capture_y;
    double target_fps;
    int jpeg_quality;
    int paint_over_jpeg_quality;
    bool use_paint_over_quality;
    int paint_over_trigger_frames;
    int damage_block_threshold;
    int damage_block_duration;

    CaptureSettings()
        : capture_width(1920),
          capture_height(1080),
          capture_x(0),
          capture_y(0),
          target_fps(60.0),
          jpeg_quality(85),
          paint_over_jpeg_quality(95),
          use_paint_over_quality(false),
          paint_over_trigger_frames(10),
          damage_block_threshold(15),
          damage_block_duration(30) {}

    CaptureSettings(int cw, int ch, int cx, int cy, double fps, int jq, int pojq, bool upoq,
                    int potf, int dbt, int dbd)
        : capture_width(cw),
          capture_height(ch),
          capture_x(cx),
          capture_y(cy),
          target_fps(fps),
          jpeg_quality(jq),
          paint_over_jpeg_quality(pojq),
          use_paint_over_quality(upoq),
          paint_over_trigger_frames(potf),
          damage_block_threshold(dbt),
          damage_block_duration(dbd) {}
};

struct StripeEncodeResult {
    int stripe_y_start;
    int stripe_height;
    int size;
    unsigned char* data;

    StripeEncodeResult() : stripe_y_start(0), stripe_height(0), size(0), data(nullptr) {}
    StripeEncodeResult(StripeEncodeResult&& other) noexcept;
    StripeEncodeResult& operator=(StripeEncodeResult&& other) noexcept;

private:
    StripeEncodeResult(const StripeEncodeResult&) = delete;
    StripeEncodeResult& operator=(const StripeEncodeResult&) = delete;
};


StripeEncodeResult::StripeEncodeResult(StripeEncodeResult&& other) noexcept
    : stripe_y_start(other.stripe_y_start),
      stripe_height(other.stripe_height),
      size(other.size),
      data(other.data) {
    other.stripe_y_start = 0;
    other.stripe_height = 0;
    other.size = 0;
    other.data = nullptr;
}

StripeEncodeResult& StripeEncodeResult::operator=(StripeEncodeResult&& other) noexcept {
    if (this != &other) {
        stripe_y_start = other.stripe_y_start;
        stripe_height = other.stripe_height;
        size = other.size;
        data = other.data;

        other.stripe_y_start = 0;
        other.stripe_height = 0;
        other.size = 0;
        other.data = nullptr;
    }
    return *this;
}

typedef void (*StripeCallback)(StripeEncodeResult* result, void* user_data);

StripeEncodeResult encode_stripe_jpeg(int thread_id, int stripe_y_start, int stripe_height,
                                     int width, int height, int capture_width_actual,
                                     const unsigned char* rgb_data, int rgb_data_len,
                                     int jpeg_quality, int frame_counter);

uint64_t calculate_stripe_hash(const std::vector<unsigned char>& rgb_data);

class ScreenCaptureModule {
public:
    int capture_width = 1920;
    int capture_height = 1080;
    int capture_x = 0;
    int capture_y = 0;
    double target_fps = 60.0;
    int jpeg_quality = 85;
    int paint_over_jpeg_quality = 95;
    bool use_paint_over_quality = false;
    int paint_over_trigger_frames = 10;
    int damage_block_threshold = 15;
    int damage_block_duration = 30;
    std::atomic<bool> stop_requested;
    std::thread capture_thread;
    StripeCallback stripe_callback = nullptr;
    void* user_data = nullptr;
    int frame_counter = 0;
    int encoded_frame_count = 0;
    mutable std::mutex settings_mutex;

public:
    ScreenCaptureModule() : stop_requested(false) {}

    void start_capture() {
        stop_requested = false;
        capture_thread = std::thread(&ScreenCaptureModule::capture_loop, this);
    }

    void stop_capture() {
        stop_requested = true;
        if (capture_thread.joinable()) {
            capture_thread.join();
        }
    }

    void modify_settings(const CaptureSettings& new_settings) {
        std::lock_guard<std::mutex> lock(settings_mutex);
        capture_width = new_settings.capture_width;
        capture_height = new_settings.capture_height;
        capture_x = new_settings.capture_x;
        capture_y = new_settings.capture_y;
        target_fps = new_settings.target_fps;
        jpeg_quality = new_settings.jpeg_quality;
        paint_over_jpeg_quality = new_settings.paint_over_jpeg_quality;
        use_paint_over_quality = new_settings.use_paint_over_quality;
        paint_over_trigger_frames = new_settings.paint_over_trigger_frames;
        damage_block_threshold = new_settings.damage_block_threshold;
        damage_block_duration = new_settings.damage_block_duration;
    }

    CaptureSettings get_current_settings() const {
        std::lock_guard<std::mutex> lock(settings_mutex);
        return CaptureSettings(capture_width, capture_height, capture_x, capture_y, target_fps,
                                 jpeg_quality, paint_over_jpeg_quality, use_paint_over_quality,
                                 paint_over_trigger_frames, damage_block_threshold,
                                 damage_block_duration);
    }

    void capture_loop() {
        auto start_time_loop = std::chrono::high_resolution_clock::now();
        int frame_count_loop = 0;
        std::chrono::duration<double> target_frame_duration_seconds =
            std::chrono::duration<double>(1.0 / target_fps);
        auto next_frame_time = start_time_loop + target_frame_duration_seconds;

        char* display_env = std::getenv("DISPLAY");
        const char* display_name = display_env ? display_env : ":0";
        std::cout << "Using DISPLAY: " << display_name << std::endl;

        Display* display = XOpenDisplay(display_name);
        if (!display) {
            std::cerr << "Error: Failed to open X display " << display_name << std::endl;
            return;
        }
        Window root_window = DefaultRootWindow(display);
        int screen = DefaultScreen(display);
        int screen_width = DisplayWidth(display, screen);
        int screen_height = DisplayHeight(display, screen);

        int capture_width_actual = capture_width;
        int capture_height_actual = capture_height;
        int capture_x_offset = capture_x;
        int capture_y_offset = capture_y;
        double current_target_fps = target_fps;
        int current_jpeg_quality = jpeg_quality;
        int current_paint_over_jpeg_quality = paint_over_jpeg_quality;
        bool current_use_paint_over_quality = use_paint_over_quality;
        int current_paint_over_trigger_frames = paint_over_trigger_frames;
        int current_damage_block_threshold = damage_block_threshold;
        int current_damage_block_duration = damage_block_duration;

        std::cout << "X11 Full Screen Dimensions: " << screen_width << "x" << screen_height
                  << std::endl;
        std::cout << "Capture Dimensions: " << capture_width_actual << "x" << capture_height_actual
                  << " at offset " << capture_x_offset << "," << capture_y_offset << std::endl;

        if (!XShmQueryExtension(display)) {
            std::cerr << "Error: X Shared Memory Extension not available!" << std::endl;
            XCloseDisplay(display);
            return;
        }
        std::cout << "X Shared Memory Extension available." << std::endl;

        XShmSegmentInfo shminfo;
        XImage* shm_image = nullptr;

        shm_image = XShmCreateImage(display, DefaultVisual(display, screen), DefaultDepth(display, screen),
                                     ZPixmap, nullptr, &shminfo, capture_width_actual,
                                     capture_height_actual);
        if (!shm_image) {
            std::cerr << "Error: XShmCreateImage failed" << std::endl;
            XCloseDisplay(display);
            return;
        }

        std::cout << "XImage Format:" << std::endl;
        std::cout << "  depth: " << shm_image->depth << std::endl;
        std::cout << "  bits_per_pixel: " << shm_image->bits_per_pixel << std::endl;
        std::cout << "  bytes_per_line: " << shm_image->bytes_per_line << std::endl;
        std::cout << "  format: "
                  << (shm_image->format == XYBitmap    ? "XYBitmap"
                      : (shm_image->format == XYPixmap ? "XYPixmap" : "ZPixmap"))
                  << std::endl;
        std::cout << "  byte_order: "
                  << (shm_image->byte_order == LSBFirst ? "LSBFirst" : "MSBFirst") << std::endl;

        int shmid = shmget(IPC_PRIVATE, shm_image->bytes_per_line * shm_image->height, IPC_CREAT | 0600);
        if (shmid < 0) {
            std::cerr << "Error: shmget failed" << std::endl;
            XDestroyImage(shm_image);
            XCloseDisplay(display);
            return;
        }

        shminfo.shmid = shmid;
        shminfo.shmaddr = (char*)shmat(shmid, nullptr, 0);
        if (shminfo.shmaddr == (char*)-1) {
            std::cerr << "Error: shmat failed" << std::endl;
            shmctl(shmid, IPC_RMID, 0);
            XDestroyImage(shm_image);
            XCloseDisplay(display);
            return;
        }
        shminfo.readOnly = False;

        shm_image->data = shminfo.shmaddr;
        if (!XShmAttach(display, &shminfo)) {
            std::cerr << "Error: XShmAttach failed" << std::endl;
            shmdt(shminfo.shmaddr);
            shmctl(shmid, IPC_RMID, 0);
            XDestroyImage(shm_image);
            XCloseDisplay(display);
            return;
        }
        std::cout << "XShm setup complete." << std::endl;

        int num_cores = std::thread::hardware_concurrency();
        std::cout << "CPU cores available: " << num_cores << std::endl;
        int num_stripes = num_cores;

        std::vector<uint64_t> previous_hashes(num_stripes, 0);
        std::vector<int> no_motion_frame_counts(num_stripes, 0);
        std::vector<uint64_t> last_paint_over_hashes(num_stripes, 0);
        std::vector<bool> paint_over_sent(num_stripes, false);
        std::vector<int> damage_block_counts(num_stripes, 0);
        std::vector<bool> damage_blocked(num_stripes, false);
        std::vector<int> damage_block_timer(num_stripes, 0);
        std::vector<int> current_jpeg_qualities(num_stripes);
        for (int i = 0; i < num_stripes; ++i) {
            current_jpeg_qualities[i] = current_use_paint_over_quality ? current_paint_over_jpeg_quality : current_jpeg_quality;
        }


        auto last_output_time = std::chrono::high_resolution_clock::now();
        encoded_frame_count = 0;

        while (!stop_requested) {
            auto loop_start_time = std::chrono::high_resolution_clock::now();
            if (loop_start_time >= next_frame_time) {
                std::stringstream log_stream;
                log_stream.str("");

                {
                    std::lock_guard<std::mutex> lock(settings_mutex);
                    capture_width_actual = capture_width;
                    capture_height_actual = capture_height;
                    capture_x_offset = capture_x;
                    capture_y_offset = capture_y;
                    current_target_fps = target_fps;
                    current_jpeg_quality = jpeg_quality;
                    current_paint_over_jpeg_quality = paint_over_jpeg_quality;
                    current_use_paint_over_quality = use_paint_over_quality;
                    current_paint_over_trigger_frames = paint_over_trigger_frames;
                    current_damage_block_threshold = damage_block_threshold;
                    current_damage_block_duration = damage_block_duration;
                    target_frame_duration_seconds = std::chrono::duration<double>(1.0 / current_target_fps);
                }

                if (XShmGetImage(display, root_window, shm_image, capture_x_offset, capture_y_offset,
                                 AllPlanes)) {
                    std::vector<unsigned char> rgb_data(capture_width_actual * capture_height_actual * 3);
                    unsigned char* shm_data = (unsigned char*)shm_image->data;
                    int bytes_per_pixel = shm_image->bits_per_pixel / 8;
                    int bytes_per_line = shm_image->bytes_per_line;

                    for (int y = 0; y < capture_height_actual; ++y) {
                        for (int x = 0; x < capture_width_actual; ++x) {
                            unsigned char* pixel_ptr =
                                shm_data + (y * bytes_per_line) + (x * bytes_per_pixel);
                            rgb_data[(y * capture_width_actual + x) * 3 + 0] = pixel_ptr[2];
                            rgb_data[(y * capture_width_actual + x) * 3 + 1] = pixel_ptr[1];
                            rgb_data[(y * capture_width_actual + x) * 3 + 2] = pixel_ptr[0];
                        }
                    }

                    std::vector<std::future<StripeEncodeResult>> futures;
                    std::vector<std::thread> threads;
                    int base_stripe_height = capture_height_actual / num_stripes;
                    int remainder_height = capture_height_actual % num_stripes;

                    std::vector<uint64_t> current_hashes(num_stripes);
                    bool frame_encoded = false;

                    for (int i = 0; i < num_stripes; ++i) {
                        int start_y = i * base_stripe_height;
                        int current_stripe_height = base_stripe_height;
                        if (i == num_stripes - 1) {
                            current_stripe_height += remainder_height;
                        }
                        if (current_stripe_height >= 0) {
                            std::vector<unsigned char> stripe_rgb_data(
                                capture_width_actual * current_stripe_height * 3);
                            int row_stride = capture_width_actual * 3;
                            for (int y_offset = 0; y_offset < current_stripe_height; ++y_offset) {
                                int global_y = start_y + y_offset;
                                if (global_y < capture_height_actual) {
                                    std::memcpy(&stripe_rgb_data[y_offset * row_stride],
                                                 &rgb_data[global_y * row_stride], row_stride);
                                } else {
                                    std::memset(&stripe_rgb_data[y_offset * row_stride], 0,
                                                 row_stride);
                                }
                            }

                            uint64_t current_hash = calculate_stripe_hash(stripe_rgb_data);
                            current_hashes[i] = current_hash;

                            if (current_hash == previous_hashes[i]) {
                                no_motion_frame_counts[i]++;
                                if (no_motion_frame_counts[i] >= current_paint_over_trigger_frames &&
                                    current_use_paint_over_quality && !paint_over_sent[i] &&
                                    !damage_blocked[i]) {
                                    std::packaged_task<StripeEncodeResult(
                                        int, int, int, int, int, int, const unsigned char*, int,
                                        int, int)>
                                        task(encode_stripe_jpeg);
                                    futures.push_back(task.get_future());
                                    threads.push_back(std::thread(
                                        std::move(task), i, start_y, current_stripe_height,
                                        screen_width, capture_height_actual, capture_width_actual,
                                        rgb_data.data(), static_cast<int>(rgb_data.size()),
                                        current_paint_over_jpeg_quality, frame_counter));
                                    last_paint_over_hashes[i] = current_hash;
                                    paint_over_sent[i] = true;
                                    frame_encoded = true;
                                }
                            } else {
                                no_motion_frame_counts[i] = 0;
                                paint_over_sent[i] = false;

                                // Decrement quality, but not below jpeg_quality
                                current_jpeg_qualities[i] = std::max(current_jpeg_qualities[i] - 1, current_jpeg_quality);

                                std::packaged_task<StripeEncodeResult(
                                    int, int, int, int, int, int, const unsigned char*, int, int, int)>
                                    task(encode_stripe_jpeg);
                                futures.push_back(task.get_future());
                                threads.push_back(std::thread(
                                    std::move(task), i, start_y, current_stripe_height, screen_width,
                                    capture_height_actual, capture_width_actual, rgb_data.data(),
                                    static_cast<int>(rgb_data.size()), current_jpeg_qualities[i], frame_counter)); 
                                previous_hashes[i] = current_hash;
                                damage_block_counts[i]++;
                                if (damage_block_counts[i] >= current_damage_block_threshold) {
                                    damage_blocked[i] = true;
                                    damage_block_timer[i] = current_damage_block_duration;
                                }
                                frame_encoded = true;
                            }
                        }
                        if (damage_block_timer[i] > 0) {
                            damage_block_timer[i]--;
                            if (damage_block_timer[i] == 0) {
                                damage_blocked[i] = false;
                                damage_block_counts[i] = 0;
                                current_jpeg_qualities[i] = current_use_paint_over_quality ? current_paint_over_jpeg_quality : current_jpeg_quality;
                            }
                        }
                    }

                    std::vector<StripeEncodeResult> stripe_results;
                    for (auto& future : futures) {
                        stripe_results.push_back(future.get());
                    }
                    futures.clear();
                    for (StripeEncodeResult& result : stripe_results) {
                        if (stripe_callback != nullptr) {
                            stripe_callback(&result, user_data);
                        }
                    }


                    for (auto& thread : threads) {
                        if (thread.joinable()) {
                            thread.join();
                        }
                    }
                    threads.clear();

                    frame_counter++;
                    if (frame_encoded) {
                        encoded_frame_count++;
                    }
                    frame_count_loop++;
                    auto current_time = std::chrono::high_resolution_clock::now();
                    auto elapsed_time =
                        std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time_loop);

                    if (elapsed_time.count() >= 1) {
                        double fps = static_cast<double>(frame_count_loop) / elapsed_time.count();
                        frame_count_loop = 0;
                        start_time_loop = std::chrono::high_resolution_clock::now();
                    }
                    next_frame_time += target_frame_duration_seconds;

                    auto current_output_time = std::chrono::high_resolution_clock::now();
                    auto output_elapsed_time =
                        std::chrono::duration_cast<std::chrono::seconds>(current_output_time - last_output_time);

                    if (output_elapsed_time.count() >= 1) {
                        double actual_fps = static_cast<double>(encoded_frame_count) / output_elapsed_time.count();
                        std::cout << "Resolution: " << capture_width_actual << "x"
                                  << capture_height_actual << " Cores/Stripes: " << num_stripes
                                  << " Actual FPS: " << std::fixed << std::setprecision(2) << actual_fps
                                  << std::endl;
                        encoded_frame_count = 0;
                        last_output_time = current_output_time;
                    }
                } else {
                    std::cerr << "Failed to capture XImage using XShmGetImage" << std::endl;
                }
            }

            auto current_loop_end_time = std::chrono::high_resolution_clock::now();
            auto time_to_sleep = next_frame_time - current_loop_end_time;
            if (time_to_sleep > std::chrono::milliseconds(0)) {
                std::this_thread::sleep_for(time_to_sleep);
            }
        }

        XShmDetach(display, &shminfo);
        shmdt(shminfo.shmaddr);
        shmctl(shmid, IPC_RMID, 0);
        XDestroyImage(shm_image);
        XCloseDisplay(display);
        std::cout << "Capture loop stopped." << std::endl;
    }
};

extern "C" {

typedef void* ScreenCaptureModuleHandle;
typedef StripeCallback StripeCallback;
typedef CaptureSettings CaptureSettings;

ScreenCaptureModuleHandle create_screen_capture_module() {
    return static_cast<ScreenCaptureModuleHandle>(new ScreenCaptureModule());
}

void destroy_screen_capture_module(ScreenCaptureModuleHandle module_handle) {
    if (module_handle) {
        delete static_cast<ScreenCaptureModule*>(module_handle);
    }
}

void start_screen_capture(ScreenCaptureModuleHandle module_handle, CaptureSettings settings,
                             StripeCallback callback, void* user_data) {
    if (module_handle) {
        ScreenCaptureModule* module = static_cast<ScreenCaptureModule*>(module_handle);
        std::lock_guard<std::mutex> lock(module->settings_mutex);
        module->capture_width = settings.capture_width;
        module->capture_height = settings.capture_height;
        module->capture_x = settings.capture_x;
        module->capture_y = settings.capture_y;
        module->target_fps = settings.target_fps;
        module->jpeg_quality = settings.jpeg_quality;
        module->paint_over_jpeg_quality = settings.paint_over_jpeg_quality;
        module->use_paint_over_quality = settings.use_paint_over_quality;
        module->paint_over_trigger_frames = settings.paint_over_trigger_frames;
        module->damage_block_threshold = settings.damage_block_threshold;
        module->damage_block_duration = settings.damage_block_duration;
        module->stripe_callback = callback;
        module->user_data = user_data;
        module->start_capture();
    }
}

void stop_screen_capture(ScreenCaptureModuleHandle module_handle) {
    if (module_handle) {
        static_cast<ScreenCaptureModule*>(module_handle)->stop_capture();
    }
}

void modify_screen_capture(ScreenCaptureModuleHandle module_handle, CaptureSettings settings) {
    if (module_handle) {
        static_cast<ScreenCaptureModule*>(module_handle)->modify_settings(settings);
    }
}

CaptureSettings get_screen_capture_settings(ScreenCaptureModuleHandle module_handle) {
    if (module_handle) {
        return static_cast<ScreenCaptureModule*>(module_handle)->get_current_settings();
    } else {
        return CaptureSettings{};
    }
}

void free_stripe_encode_result_data(StripeEncodeResult* result) {
    if (result && result->data) {
        delete[] result->data;
        result->data = nullptr;
        result->size = 0;
    }
}


} // extern "C"


StripeEncodeResult encode_stripe_jpeg(int thread_id, int stripe_y_start, int stripe_height,
                                     int width, int height, int capture_width_actual,
                                     const unsigned char* rgb_data, int rgb_data_len,
                                     int jpeg_quality, int frame_counter) {
    StripeEncodeResult result;
    result.stripe_y_start = stripe_y_start;
    result.stripe_height = stripe_height;

    jpeg_compress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    cinfo.image_width = capture_width_actual;
    cinfo.image_height = stripe_height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, jpeg_quality, TRUE);

    unsigned char* jpeg_buffer = nullptr;
    unsigned long jpeg_size_temp = 0;
    jpeg_mem_dest(&cinfo, &jpeg_buffer, &jpeg_size_temp);
    jpeg_start_compress(&cinfo, TRUE);

    JSAMPROW row_pointer[1];
    int row_stride = capture_width_actual * 3;
    std::vector<unsigned char> stripe_rgb_data(capture_width_actual * stripe_height * 3);

    for (int y_offset = 0; y_offset < stripe_height; ++y_offset) {
        int global_y = stripe_y_start + y_offset;
        if (global_y < height) {
            if (rgb_data != nullptr) {
                std::memcpy(&stripe_rgb_data[y_offset * row_stride], &rgb_data[global_y * row_stride],
                             row_stride);
            } else {
                std::memset(&stripe_rgb_data[y_offset * row_stride], 0, row_stride);
            }
        } else {
            std::memset(&stripe_rgb_data[y_offset * row_stride], 0, row_stride);
        }
        row_pointer[0] = &stripe_rgb_data[y_offset * row_stride];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);

    int padding_size = 4;
    unsigned char* padded_jpeg_buffer = new unsigned char[jpeg_size_temp + padding_size];
    uint16_t frame_counter_net = htons(static_cast<uint16_t>(frame_counter));
    uint16_t stripe_y_start_net = htons(static_cast<uint16_t>(stripe_y_start));

    std::memcpy(padded_jpeg_buffer, &frame_counter_net, 2);
    std::memcpy(padded_jpeg_buffer + 2, &stripe_y_start_net, 2);
    std::memcpy(padded_jpeg_buffer + padding_size, jpeg_buffer, jpeg_size_temp);

    result.size = static_cast<int>(jpeg_size_temp) + padding_size;
    result.data = padded_jpeg_buffer;
    jpeg_destroy_compress(&cinfo);
    free(jpeg_buffer);
    return result;
}

uint64_t calculate_stripe_hash(const std::vector<unsigned char>& rgb_data) {
    return XXH3_64bits(rgb_data.data(), rgb_data.size());
}
