@echo off
color 0a
echo ========================================
echo    LogHuntEnv - Cybersecurity AI
echo ========================================
echo.
echo Installing libraries...
pip install numpy pandas scikit-learn gymnasium stable-baselines3 matplotlib --quiet
echo Done!
echo.
echo Creating dataset...
python create_dataset.py
echo.
echo Starting AI...
echo This takes 5-10 minutes, dont close this window!
echo.
python train_ppo.py
echo.
echo ========================================
echo    FINISHED! Check results above!
echo ========================================
pause
