import os
import shutil

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(base_dir, "backend")

    # Create backend directory if it doesn't exist
    if not os.path.exists(backend_dir):
        os.makedirs(backend_dir)
        print(f"✅ Created directory: {backend_dir}")
    else:
        print(f"ℹ️ Directory already exists: {backend_dir}")

    # Files and folders to move
    items_to_move = [
        "app.py",
        "database.py",
        "recognize_attendance.py",
        "capture_faces.py",
        "requirements.txt",
        "attendance.db",
        "system.log",
        "system_logs.jsonl",
        "TrainingImage",
        "templates",
        "static"
    ]

    moved_count = 0

    print("\n📦 Moving backend files...")
    for item in items_to_move:
        source_path = os.path.join(base_dir, item)
        dest_path = os.path.join(backend_dir, item)

        if os.path.exists(source_path):
            try:
                shutil.move(source_path, dest_path)
                print(f"  ➡️ Moved: {item}")
                moved_count += 1
            except Exception as e:
                print(f"  ❌ Error moving {item}: {e}")
        else:
            print(f"  ⚠️ Skipping {item} (Not found in root folder)")

    print(f"\n🎉 Successfully moved {moved_count} items to the backend folder!")
    print("\nNext Steps:")
    print("1. Open your terminal and navigate to the backend folder:")
    print("   cd backend")
    print("2. Run your application as usual:")
    print("   python app.py")

if __name__ == "__main__":
    main()
