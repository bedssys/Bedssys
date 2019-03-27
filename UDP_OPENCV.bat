@ECHO OFF
ECHO Must be run with Administrator Right!
ECHO.
ECHO Somehow, this only works for CMD and not for Powershell.
ECHO Thus you must run the python .py from CMD, for now.
ECHO.
setx OPENCV_FFMPEG_CAPTURE_OPTIONS "rtsp_transport;udp" /m
pause