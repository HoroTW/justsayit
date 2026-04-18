#!/bin/sh
set -eu

local_time=$(date +%H:%M:%S)
local_date=$(date +%F)
timezone=""

if [ -L /etc/localtime ]; then
    localtime_target=$(readlink /etc/localtime 2>/dev/null || true)
    case "$localtime_target" in
        *zoneinfo/*)
            timezone=${localtime_target##*zoneinfo/}
            ;;
    esac
fi

if [ -z "$timezone" ] && [ -r /etc/timezone ]; then
    IFS= read -r timezone < /etc/timezone || true
fi

if [ -z "$timezone" ]; then
    timezone=$(date +%Z)
fi

printf 'Local time: %s\n' "$local_time"
printf 'Date: %s\n' "$local_date"
printf 'Timezone: %s\n' "$timezone"

locale_hint=${LC_ALL:-${LC_TIME:-${LANG:-}}}
locale_hint=${locale_hint%%.*}
if [ -n "$locale_hint" ] && [ "$locale_hint" != "C" ] && [ "$locale_hint" != "POSIX" ]; then
    printf 'Locale hint: %s\n' "$locale_hint"
fi
