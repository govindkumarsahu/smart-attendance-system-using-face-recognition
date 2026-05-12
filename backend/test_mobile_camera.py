"""
Mobile Camera Complete Diagnostic
Run: python test_mobile_camera.py
"""
import cv2
import urllib.request
import sys

# ====== APNA IP YAHAN DAALO ======
MOBILE_IP = "192.168.43.1"   # IP Webcam app me jo dikh raha hai
MOBILE_PORT = "8080"
# ==================================

URL_BASE  = f"http://{MOBILE_IP}:{MOBILE_PORT}"
URL_VIDEO = f"{URL_BASE}/video"

print("=" * 50)
print("  MOBILE CAMERA DIAGNOSTIC TOOL")
print("=" * 50)
print(f"\n[1] Testing HTTP ping to {URL_BASE} ...")
try:
    req = urllib.request.urlopen(URL_BASE, timeout=4)
    print(f"    ✅ HTTP REACHABLE — status {req.status}")
    print(f"    ✅ Laptop CAN see the mobile on network")
except Exception as e:
    print(f"    ❌ HTTP FAILED: {e}")
    print()
    print("    REASON: Mobile aur Laptop SAME WiFi pe nahi hain")
    print("    FIX OPTIONS:")
    print("    Option A: Mobile ka HOTSPOT on karo → Laptop usse connect karo")
    print("    Option B: Dono ko SAME WiFi (college/home) se connect karo")
    print()
    print("    IP Webcam app band karo → Hotspot on karo → phir Start server")
    sys.exit(1)

print(f"\n[2] Testing video stream at {URL_VIDEO} ...")
cap = cv2.VideoCapture(URL_VIDEO)
if cap.isOpened():
    ret, frame = cap.read()
    if ret and frame is not None:
        print("    ✅ VIDEO STREAM WORKING!")
        print("    ✅ Mobile camera BILKUL ready hai attendance ke liye!")
        print("\n    Opening preview... Press Q to quit")
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow("Mobile Camera - LIVE", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("    ⚠️  Camera connected but NO FRAMES received")
        print("    FIX: IP Webcam app → Settings → Video → lower resolution try karo")
else:
    print("    ❌ VIDEO stream failed even though HTTP works")
    print(f"    Try this URL instead in Rooms: {URL_BASE}/videofeed")
    print(f"    Or try: {URL_BASE}/shot.jpg")

print("\n" + "=" * 50)
print(f"  URL to put in Admin → Rooms: {URL_VIDEO}")
print("=" * 50)
