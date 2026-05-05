#!/bin/bash
gunicorn --bind=0.0.0.0 --timeout 3600 --workers 4 --threads 2 main:app
