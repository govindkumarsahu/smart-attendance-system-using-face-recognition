@echo off
echo Cleaning up obsolete legacy Computer Vision files...

del /f /q train_model.py
del /f /q train_model_fixed.py
del /f /q train_model_improved.py
del /f /q recognize_and_attendance.py
del /f /q recognize_and_attendance_improved.py
del /f /q recognize_face.py
del /f /q dataset_capture.py
del /f /q trainer.yml
del /f /q labels.npy
del /f /q haarcascade_frontalface_default.xml

echo Renaming dataset to TrainingImage...
ren dataset TrainingImage

echo Cleanup complete!
pause
