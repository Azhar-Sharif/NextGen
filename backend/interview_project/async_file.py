import asyncio

def run_async(coro):
    """Runs an asynchronous coroutine in the current event loop."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:  # If no event loop is running
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)