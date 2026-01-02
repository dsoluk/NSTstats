from django.core.management.base import BaseCommand
import sys
import os
from app.cli import main as cli_main

class Command(BaseCommand):
    help = 'Run NSTstats CLI commands'

    def add_arguments(self, parser):
        # We want to pass all arguments to the original CLI main
        parser.add_argument('cli_args', nargs='*', help='Arguments for the original CLI')

    def handle(self, *args, **options):
        # Reconstruct sys.argv for the original CLI
        original_argv = sys.argv
        # sys.argv[0] is 'manage.py', sys.argv[1] is 'nststats'
        # we want to pass the rest
        new_argv = ['nststats'] + options['cli_args']
        
        # Override sys.argv and call original main
        sys.argv = new_argv
        try:
            cli_main()
        except SystemExit:
            pass
        finally:
            sys.argv = original_argv
