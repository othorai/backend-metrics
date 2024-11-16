# deployment/scripts/load-env.sh
#!/bin/bash

# Function to load .env file
load_env() {
    if [ -f .env ]; then
        export $(cat .env | sed 's/#.*//g' | xargs)
    else
        echo ".env file not found"
        exit 1
    fi
}