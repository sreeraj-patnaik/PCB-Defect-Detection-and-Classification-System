import cv2
import numpy as np
from model import predict


# --------------------------------------------------
# Image Alignment using ORB Feature Matching
# --------------------------------------------------
def align_images(template, defect):

    orb = cv2.ORB_create(5000)

    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(defect, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)

    aligned = cv2.warpPerspective(
        defect,
        H,
        (template.shape[1], template.shape[0])
    )

    return aligned


# --------------------------------------------------
# ROI Extraction using Difference Detection
# --------------------------------------------------
def extract_rois(template_path, defect_path):

    template = cv2.imread(template_path)
    defect = cv2.imread(defect_path)

    aligned_defect = align_images(template, defect)

    # Absolute difference between template and aligned defect
    diff = cv2.absdiff(template, aligned_defect)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Normalize to amplify faint differences
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Low threshold to capture subtle defects
    _, thresh = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)

    # Remove small noise
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Slight dilation to group defect pixels
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Debug images
    cv2.imwrite("uploads/debug_diff.png", diff)
    cv2.imwrite("uploads/debug_gray.png", gray)
    cv2.imwrite("uploads/debug_thresh.png", thresh)

    # Find contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    rois = []

    for i, cnt in enumerate(contours):

        x, y, w, h = cv2.boundingRect(cnt)

        # Ignore tiny noise
        if w * h < 500:
            continue

        # Add padding around defect
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(aligned_defect.shape[1], x + w + pad)
        y2 = min(aligned_defect.shape[0], y + h + pad)

        roi = aligned_defect[y1:y2, x1:x2]

        roi_path = f"uploads/roi_{i}.png"
        cv2.imwrite(roi_path, roi)

        rois.append(roi_path)

    return rois


# --------------------------------------------------
# Full Processing Pipeline
# --------------------------------------------------
def process_images(template_path, defect_path):

    roi_paths = extract_rois(template_path, defect_path)

    results = []

    for roi in roi_paths:

        label, confidence = predict(roi)

        results.append({
            "roi": roi,
            "prediction": label,
            "confidence": round(confidence * 100, 2)
        })

    return results