use std::ffi::c_void;
use libloading::{Library, Symbol};
use std::fs::File;
use std::io::Write;

#[repr(C)]
pub struct CaptureSettings {
    pub capture_width: i32,
    pub capture_height: i32,
    pub capture_x: i32,
    pub capture_y: i32,
    pub target_fps: f64,
    pub jpeg_quality: i32,
    pub paint_over_jpeg_quality: i32,
    pub use_paint_over_quality: bool,
    pub paint_over_trigger_frames: i32,
    pub damage_block_threshold: i32,
    pub damage_block_duration: i32,
}

#[repr(C)]
pub struct StripeEncodeResult {
    pub stripe_y_start: i32,
    pub stripe_height: i32,
    pub jpeg_size: i32,
    pub jpeg_data: *mut u8,
}

pub type StripeCallback = extern "C" fn(result: *mut StripeEncodeResult);


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lib = unsafe { Library::new("./libscreencapturemodule.so")? };

    type CreateModuleFn = unsafe extern "C" fn() -> *mut c_void;
    type DestroyModuleFn = unsafe extern "C" fn(module: *mut c_void);
    type StartCaptureFn = unsafe extern "C" fn(
        module: *mut c_void,
        settings: CaptureSettings,
        callback: StripeCallback,
    );
    type StopCaptureFn = unsafe extern "C" fn(module: *mut c_void);

    let create_module: Symbol<CreateModuleFn> = unsafe { lib.get(b"create_screen_capture_module\0")? };
    let destroy_module: Symbol<DestroyModuleFn> = unsafe { lib.get(b"destroy_screen_capture_module\0")? };
    let start_capture: Symbol<StartCaptureFn> = unsafe { lib.get(b"start_screen_capture\0")? };
    let stop_capture: Symbol<StopCaptureFn> = unsafe { lib.get(b"stop_screen_capture\0")? };

    extern "C" fn callback_handler(result_ptr: *mut StripeEncodeResult) {
        if !result_ptr.is_null() {
            let result = unsafe { &*result_ptr };
            println!(
                "Rust Callback received stripe: start_y={}, height={}, jpeg_size={} bytes",
                result.stripe_y_start, result.stripe_height, result.jpeg_size
            );

            if result.jpeg_size > 0 && !result.jpeg_data.is_null() {
                unsafe {
                    let jpeg_slice = std::slice::from_raw_parts(result.jpeg_data, result.jpeg_size as usize);

                    match File::create("output.jpeg") {
                        Ok(mut file) => {
                            match file.write_all(jpeg_slice) {
                                Ok(_) => println!("JPEG data written to output.jpeg"),
                                Err(e) => eprintln!("Error writing JPEG data to file: {}", e),
                            }
                        }
                        Err(e) => eprintln!("Error creating file output.jpeg: {}", e),
                    }
                }
            } else {
                println!("No JPEG data to write (size is zero or data pointer is null).");
            }
        }
    }


    unsafe {
        let module_handle = create_module();
        if module_handle.is_null() {
            eprintln!("Failed to create screen capture module");
            return Ok(());
        }

        let settings = CaptureSettings {
            capture_width: 1920,
            capture_height: 1080,
            capture_x: 0,
            capture_y: 0,
            target_fps: 30.0,
            jpeg_quality: 80,
            paint_over_jpeg_quality: 90,
            use_paint_over_quality: false,
            paint_over_trigger_frames: 10,
            damage_block_threshold: 15,
            damage_block_duration: 30,
        };

        println!("Starting screen capture...");
        start_capture(module_handle, settings, callback_handler);

        println!("Sleeping for 5 seconds...");
        std::thread::sleep(std::time::Duration::from_secs(5));

        println!("Stopping screen capture...");
        stop_capture(module_handle);

        println!("Destroying screen capture module...");
        destroy_module(module_handle);
        println!("Screen capture module destroyed.");
    }

    println!("Rust main finished.");
    Ok(())
}
