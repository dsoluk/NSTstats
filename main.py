import sys
from dotenv import load_dotenv
from app import main as _app_main










if __name__ == '__main__':
    # Ensure .env values are loaded before running the app
    _app_main()
    sys.exit(0)
