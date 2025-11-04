import asyncio
from multi_agent import run_multi_agent

async def main():
    await run_multi_agent()


if __name__ == "__main__":
    asyncio.run(main())
