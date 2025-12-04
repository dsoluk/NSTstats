import os
from dotenv import load_dotenv
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# This Alembic env.py uses DATABASE_URL from environment and the project models' metadata
# Load .env so CLI invocations like `alembic upgrade head` pick up local settings
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/app.db")

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

from infrastructure.persistence import Base  # noqa: E402
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = DATABASE_URL
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        {"sqlalchemy.url": DATABASE_URL},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        # Enable SQLite batch mode to allow constraint/index changes via copy-and-move
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
