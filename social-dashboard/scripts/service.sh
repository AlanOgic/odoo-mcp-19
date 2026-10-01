#!/bin/zsh
# Runs the planner as a macOS launch agent: it starts when you log in, restarts
# if it crashes, and serves the last build on the local network (port 3100).
#
#   npm run service:install     build, then install and start the agent
#   npm run service:restart     rebuild and restart (after code changes)
#   npm run service:uninstall   stop it and remove the agent
#   npm run service:status      is it running?
#
# Logs: ~/Library/Logs/studiostone-social-planner.log
set -euo pipefail

LABEL="com.studiostone.social-planner"
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/studiostone-social-planner.log"
DOMAIN="gui/$(id -u)"
NODE_BIN="$(dirname "$(command -v node)")"

write_plist() {
  mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
  cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>WorkingDirectory</key><string>$APP_DIR</string>
  <key>ProgramArguments</key>
  <array>
    <string>$NODE_BIN/node</string>
    <string>$APP_DIR/node_modules/next/dist/bin/next</string>
    <string>start</string><string>-p</string><string>3100</string><string>-H</string><string>0.0.0.0</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>$NODE_BIN:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>NODE_ENV</key><string>production</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict>
</plist>
EOF
}

build() {
  (cd "$APP_DIR" && "$NODE_BIN/npm" run build)
}

case "${1:-}" in
  install)
    build
    write_plist
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    launchctl bootstrap "$DOMAIN" "$PLIST"
    echo "Installed. The planner starts at login on port 3100. Logs: $LOG"
    ;;
  restart)
    build
    launchctl kickstart -k "$DOMAIN/$LABEL"
    echo "Rebuilt and restarted."
    ;;
  uninstall)
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
    echo "Removed. The planner no longer starts at login."
    ;;
  status)
    launchctl print "$DOMAIN/$LABEL" 2>/dev/null | grep -E "^\s*(state|pid|last exit code) =" || echo "Not installed."
    ;;
  *)
    echo "Usage: service.sh install | restart | uninstall | status" >&2
    exit 1
    ;;
esac
