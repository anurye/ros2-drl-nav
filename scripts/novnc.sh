#!/bin/bash
# Start a virtual X desktop (:1) and serve it at http://localhost:6080/vnc.html
# Run inside the container, then launch GUIs with DISPLAY=:1.
set -e
PORT=${NOVNC_PORT:-6080}
GEOMETRY=${VNC_GEOMETRY:-1600x1000}
vncserver -kill :1 >/dev/null 2>&1 || true
vncserver :1 -geometry "$GEOMETRY" -depth 24 -localhost no -SecurityTypes None --I-KNOW-THIS-IS-INSECURE \
  -xstartup /usr/bin/openbox-session >/tmp/vncserver.log 2>&1
websockify --daemon --web /usr/share/novnc "$PORT" localhost:5901 >/tmp/websockify.log 2>&1
echo "Desktop ready: http://localhost:$PORT/vnc.html  (use DISPLAY=:1)"
