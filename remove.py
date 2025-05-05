import cv2
import numpy as np
import onnxruntime as ort
import os
from io import BytesIO


def remove(image_input, input_size=(320, 320), return_mask=False, top_margin_percent=0.1, dpi=300):

    try:
        # تحميل النموذجين - مسارات ثابتة
        u2net_path = r"C:\SnapSheet\models\u2net.onnx"
        modnet_path = r"C:\SnapSheet\models\modnet_photographic_portrait_matting.onnx"
        
        # التحقق من وجود الملفات
        if not os.path.isfile(u2net_path):
            raise FileNotFoundError(f"Model file not found: {u2net_path}")
        
        if not os.path.isfile(modnet_path):
            raise FileNotFoundError(f"Model file not found: {modnet_path}")

        # إنشاء جلسات ONNX
        u2net_session = ort.InferenceSession(u2net_path)
        modnet_session = ort.InferenceSession(modnet_path)

        # قراءة الصورة
        if isinstance(image_input, str):
            image = cv2.imread(image_input, cv2.IMREAD_UNCHANGED)
        elif isinstance(image_input, (bytes, bytearray)):
            image = cv2.imdecode(np.frombuffer(image_input, np.uint8), cv2.IMREAD_UNCHANGED)
        elif isinstance(image_input, BytesIO):
            image = cv2.imdecode(np.frombuffer(image_input.getvalue(), np.uint8), cv2.IMREAD_UNCHANGED)
        else:
            raise ValueError("Unsupported input type. Use path, bytes, or BytesIO.")

        if image is None:
            raise ValueError("Failed to load image.")

        # حفظ الصورة الأصلية (BGR)
        original_image_bgr = image.copy()
        if original_image_bgr.ndim == 2:
            original_image_bgr = cv2.cvtColor(original_image_bgr, cv2.COLOR_GRAY2BGR)
        elif original_image_bgr.shape[2] == 4:
            original_image_bgr = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGRA2BGR)

        # تحويل الصورة لـ RGB للمعالجة
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_h, original_w = image.shape[:2]

        # ===== المعالجة باستخدام U2Net =====
        resized = cv2.resize(image, input_size)
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # تشغيل U2Net
        u2net_output = u2net_session.run(
            [u2net_session.get_outputs()[0].name],
            {u2net_session.get_inputs()[0].name: input_tensor}
        )[0]

        mask = u2net_output.squeeze()
        mask = cv2.resize(mask, (original_w, original_h))
        mask = (mask * 255).astype(np.uint8)

        # ===== المعالجة باستخدام MODNet =====
        # تحضير الصورة لـ MODNet (512x512)
        modnet_size = (512, 512)
        resized_modnet = cv2.resize(image, modnet_size)
        resized_modnet = resized_modnet.astype(np.float32) / 255.0
        input_tensor_modnet = np.transpose(resized_modnet, (2, 0, 1))
        input_tensor_modnet = np.expand_dims(input_tensor_modnet, axis=0)

        # تشغيل MODNet
        modnet_output = modnet_session.run(
            [modnet_session.get_outputs()[0].name],
            {modnet_session.get_inputs()[0].name: input_tensor_modnet}
        )[0]

        modnet_mask = modnet_output.squeeze()
        modnet_mask = cv2.resize(modnet_mask, (original_w, original_h))
        modnet_mask = (modnet_mask * 255).astype(np.uint8)

        # ===== دمج النتائج للحصول على أفضل جودة =====
        # تطبيق threshold على كلا القناعين
        _, u2net_thresh = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        _, modnet_thresh = cv2.threshold(modnet_mask, 128, 255, cv2.THRESH_BINARY)
        
        # دمج القناعين مع إعطاء أولوية لـ MODNet في منطقة الوجه
        combined_mask = cv2.bitwise_or(u2net_thresh, modnet_thresh)
        
        # تحسين الحواف باستخدام morphological operations
        kernel = np.ones((3,3), np.uint8)
        refined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
        
        # تطبيق Gaussian blur لنعومة الحواف
        final_mask = cv2.GaussianBlur(refined_mask, (5,5), 0)

        if return_mask:
            success, buf = cv2.imencode('.png', final_mask)
            if not success:
                raise ValueError("Failed to encode mask.")
            return buf.tobytes()

        # إنشاء الصورة النهائية مع قناة ألفا
        result_image = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2BGRA)
        result_image[:, :, 3] = final_mask

        # تحديد حدود الشخص
        _, thresh = cv2.threshold(final_mask, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("No person detected in the image")

        # الحصول على أكبر كونتور (الشخص)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # حساب المساحة الإضافية (10% من ارتفاع الصورة الأصلية)
        extra_space = int(original_h * top_margin_percent)
        
        # إنشاء صورة جديدة مع المساحة الإضافية
        new_height = h + extra_space
        new_width = w
        
        # إنشاء صورة فارغة مع قناة ألفا
        final_image = np.zeros((new_height, new_width, 4), dtype=np.uint8)
        
        # وضع الشخص في الجزء السفلي من الصورة الجديدة مع ترك المساحة الإضافية في الأعلى
        final_image[extra_space:, :] = result_image[y:y+h, x:x+w]

        # تحويل الصورة النهائية إلى bytes
        success, output_buffer = cv2.imencode(".png", final_image)
        if not success:
            raise ValueError("Failed to encode output image.")

        return output_buffer.tobytes()

    except Exception as e:
        print(f"[remove] Error: {e}")
        raise


# image_bytes = remove(image_uri) 
# np_array = np.frombuffer(image_bytes, dtype=np.uint8)  
# image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)  
