REM verify redis is running already (user1@MSI:/$ sudo redis-server /etc/redis/redis.conf)
call .\venv\Scripts\activate.bat
rem call .\venv_testing\Scripts\activate.bat
REM copy /b/v/y "C:\Users\kchin\Documents\My Games\Rocket League\TAGame\Config\TASystemSettings_jeff.ini" "C:\Users\kchin\Documents\My Games\Rocket League\TAGame\Config\TASystemSettings.ini"
REM copy /b/v/y "C:\Users\kchin\AppData\Roaming\bakkesmod\bakkesmod\cfg\plugins_bots.cfg" "C:\Users\kchin\AppData\Roaming\bakkesmod\bakkesmod\cfg\plugins.cfg"
REM copy /b /v /y "C:\Users\kchin\Documents\My Games\Rocket League\TAGame\Config\TASystemSettings_bots.ini" "C:\Users\kchin\Documents\My Games\Rocket League\TAGame\Config\TASystemSettings.ini"
CHOICE /C YN /M "Start learner?"
IF ERRORLEVEL 2 goto loop
IF ERRORLEVEL 1 GOTO learner

:learner
REM start python learner_gp.py
start "LEARNER" cmd /k "call .\venv\Scripts\activate.bat & python learner.py"
TIMEOUT 15
REM start python -m training.worker 1 localhost MSI STREAMER
:loop
REM start python worker_gp.py localhost MSI GAMESTATE
REM start "CHECK" cmd /k "call .\venv\Scripts\activate.bat & python worker_gp.py localhost MSI GAMESTATE"
REM TIMEOUT 30
REM FOR /L %%G IN (1,1,1) DO (start "CHECK" cmd /k "call .\venv\Scripts\activate.bat & python worker.py localhost MSI GAMESTATE" & TIMEOUT 4)
FOR /L %%G IN (1,1,1) DO (start "CHECK" cmd /k "call .\venv\Scripts\activate.bat & python worker.py localhost MSI" & TIMEOUT 4)
REM FOR /L %%G IN (1,1,14) DO (start python worker.py localhost MSI GAMESTATE & TIMEOUT 4)
FOR /L %%G IN (1,1,14) DO (start python worker.py localhost MSI & TIMEOUT 4)
REM :workers
REM start python worker_gp.py localhost MSI GAMESTATE
REM TIMEOUT 2
REM start python worker_gp.py localhost MSI
REM TIMEOUT 2
REM start python worker_gp.py localhost MSI
REM TIMEOUT 2
REM start python worker_gp.py localhost MSI
REM TIMEOUT 2
REM start python worker_gp.py localhost MSI
REM TIMEOUT 2
REM start python worker_gp.py localhost MSI 
REM TIMEOUT 2
REM start python worker_gp.py localhost MSI
REM TIMEOUT 2
REM start python worker_gp.py localhost MSI
REM TIMEOUT 15
REM copy /b /v /y "C:\Users\kchin\Documents\My Games\Rocket League\TAGame\Config\TASystemSettings_bots.ini" "C:\Users\kchin\Documents\My Games\Rocket League\TAGame\Config\TASystemSettings.ini"
REM TIMEOUT 15
REM start python worker_gp.py localhost MSI
REM TIMEOUT 15
REM copy /b /v /y "C:\Users\kchin\Documents\My Games\Rocket League\TAGame\Config\TASystemSettings_jeff.ini" "C:\Users\kchin\Documents\My Games\Rocket League\TAGame\Config\TASystemSettings.ini"

REM copy /b/v/y "C:\Users\kchin\Documents\My Games\Rocket League\TAGame\Config\TASystemSettings_bots.ini" "C:\Users\kchin\Documents\My Games\Rocket League\TAGame\Config\TASystemSettings.ini"
REM copy /b/v/y "C:\Users\kchin\Documents\My Games\Rocket League\TAGame\Config\TASystemSettings_actual.ini" "C:\Users\kchin\Documents\My Games\Rocket League\TAGame\Config\TASystemSettings.ini"
