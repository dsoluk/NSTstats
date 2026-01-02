from django.core.management.base import BaseCommand
import sys
import os

class Command(BaseCommand):
    help = 'Interactive menu for NSTstats'

    def handle(self, *args, **options):
        while True:
            self.stdout.write("\n=== NSTstats Interactive Menu ===")
            self.stdout.write("1. Daily Workflow (sync-rosters, all, schedule-lookup)")
            self.stdout.write("2. Weekly Workflow (fetch-daily-gp, backfill-gp, avg-compare)")
            self.stdout.write("3. Run Yahoo Pipeline")
            self.stdout.write("4. Run NST Pipeline")
            self.stdout.write("5. Merge Data")
            self.stdout.write("6. Forecast")
            self.stdout.write("7. Compare")
            self.stdout.write("8. Analyze")
            self.stdout.write("9. Run Django Admin Server")
            self.stdout.write("0. Exit")
            
            choice = input("\nEnter choice: ")
            
            if choice == '1':
                os.system('python manage.py nststats daily')
            elif choice == '2':
                os.system('python manage.py nststats weekly')
            elif choice == '3':
                os.system('python manage.py nststats yahoo')
            elif choice == '4':
                os.system('python manage.py nststats nst')
            elif choice == '5':
                os.system('python manage.py nststats merge')
            elif choice == '6':
                week = input("Enter current week: ")
                os.system(f'python manage.py nststats forecast --current-week {week}')
            elif choice == '7':
                week = input("Enter current week: ")
                os.system(f'python manage.py nststats compare --current-week {week}')
            elif choice == '8':
                os.system('python manage.py nststats analyze')
            elif choice == '9':
                self.stdout.write("Starting server... Press Ctrl+C to stop and return to menu.")
                os.system('python manage.py runserver')
            elif choice == '0':
                break
            else:
                self.stdout.write(self.style.ERROR("Invalid choice"))
