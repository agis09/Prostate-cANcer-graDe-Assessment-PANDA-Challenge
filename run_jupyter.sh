#!/bin/bash
jupyter lab \
--no-browser \
--port=8888 \
--ip=0.0.0.0 \
--allow-root\
"$@"
