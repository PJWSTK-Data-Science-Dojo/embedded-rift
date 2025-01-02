import time
import httpx


class APIHandler:
    def __init__(self, rate_window: float, rate_limit: int):
        """
        Initializes the APIHandler with rate-limiting parameters.

        :param rate_window: The time window (in seconds) for rate-limiting.
        :param rate_limit: The maximum number of requests allowed within the rate_window.
        """
        self.rate_window = rate_window
        self.rate_limit = rate_limit
        self.request_times = []

    def _rate_limit_check(self):
        """
        Ensures the API requests stay within the specified rate limit.
        If the limit is exceeded, pauses execution until a request can be made.
        """
        current_time = time.time()

        # Remove timestamps outside the rate window
        self.request_times = [
            t for t in self.request_times if current_time - t < self.rate_window
        ]

        if len(self.request_times) >= self.rate_limit:
            wait_time = self.rate_window - (current_time - self.request_times[0])
            time.sleep(wait_time)

            # Re-check timestamps after waiting
            self.request_times = [
                t for t in self.request_times if time.time() - t < self.rate_window
            ]

    def _send_request(
        self, url: str, *, headers: dict = None, params: dict = None
    ) -> httpx.Response:
        """
        Sends a GET request to the specified URL and returns the response.

        :param url: The endpoint URL.
        :param headers: Optional headers for the request.
        :param params: Optional query parameters for the request.
        :return: The HTTP response object.
        """
        if headers is None:
            headers = {}

        self._rate_limit_check()  # Enforce rate limit
        response = httpx.get(url, headers=headers, params=params)

        if response.status_code == 429:
            print(f"Rate limit exceeded: {response.status_code}, {response.text}")
            time.sleep(self.rate_window)  # Wait before retrying
            response = httpx.get(url, headers=headers, params=params)

            if response.status_code == 429:
                raise Exception(
                    f"Rate limit exceeded even after waiting. {response.status_code}, {response.text}"
                )

        self.request_times.append(time.time())
        return response

    def get_json(self, url: str, *, headers: dict = None, params: dict = None) -> dict:
        """
        Sends a GET request and returns the response JSON.

        :param url: The endpoint URL.
        :param headers: Optional headers for the request.
        :param params: Optional query parameters for the request.
        :return: The response JSON as a dictionary.
        """
        response = self._send_request(url, headers=headers, params=params)
        return response.json()

    def get_content(
        self, url: str, *, headers: dict = None, params: dict = None
    ) -> str:
        """
        Sends a GET request and returns the raw response content.

        :param url: The endpoint URL.
        :param headers: Optional headers for the request.
        :param params: Optional query parameters for the request.
        :return: The raw response content.
        """
        response = self._send_request(url, headers=headers, params=params)
        return response.content.decode("utf-8")

    def send_request(
        self, url, *, headers: dict = None, params: dict = None
    ) -> httpx.Response:
        response = self._send_request(url, headers=headers, params=params)
        return response
