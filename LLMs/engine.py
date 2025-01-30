import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm
import numpy as np
import re

from openai import OpenAI

from tenacity import (
  retry,
  stop_after_attempt,
  wait_random_exponential,
)  # for exponential backoff

try:
    from google import genai
except ImportError:
    pass

from json.decoder import JSONDecodeError


class AsyncParityCalculator:
    def __init__(self, api_key: str, n_max_tokens: int):
        """
        Initialize the asynchronous OpenAI client and store the maximum tokens.
        """
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=600)
        self.n_max_tokens = n_max_tokens

    async def _compute_parity(self, bitstring: str) -> str:
        """
        Send one asynchronous request to compute the parity of 'bitstring'.
        """
        response = await self.client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You will receive a string. You have to manually calculate its parity. "
                        "Finish your response with 1 if the parity is odd, and 0 if the parity is even."
                    )
                },
                {
                    "role": "user",
                    "content": bitstring
                }
            ],
            stream=False,
            max_tokens=500
        )
        return response.choices[0].message.reasoning_content + "\n-------\n" + response.choices[0].message.content,  \
               bitstring, response.usage.completion_tokens

    async def run(self, bitstrings):
        """
        Given a list of bitstrings, schedule and gather all parity computations 
        asynchronously. Returns a list of responses corresponding to each bitstring.
        A tqdm progress bar is displayed showing number of completed requests.
        """
        # Create a list of tasks
        tasks = [asyncio.create_task(self._compute_parity(bs)) for bs in bitstrings]

        responses = []
        # Use as_completed to update the progress bar as each task finishes
        try:
            with tqdm(total=len(tasks)) as pbar:
                for coro in asyncio.as_completed(tasks, timeout=600):
                    try:
                        res = await coro
                    except JSONDecodeError as e:
                        print(f"Error: {e}")
                        continue
                    responses.append(res)
                    pbar.update(1)
        except asyncio.TimeoutError:
            print(f"Timeout reached. Returning {len(responses)} responses.")

        return responses
    

class AsyncMultiplicationCalculator:
    def __init__(self, api_key: str, n_max_tokens: int):
        """
        Initialize the asynchronous OpenAI client and store the maximum tokens.
        """
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=600)
        self.n_max_tokens = n_max_tokens

    async def _multiply(self, x, y):
        response = await self.client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You will receive two numbers. You have to multiply them manually. "
                        "Finish your response with the precise result of the multiplication."
                    )
                },
                {
                    "role": "user",
                    "content": f"{x} {y}"
                }
            ],
            stream=False,
        )
        # print(response.usage)
        return response.choices[0].message.reasoning_content + "\n-------\n" + response.choices[0].message.content,  \
               x, y, response.usage.completion_tokens

    async def run(self, tests):
        """
        Given a list of bitstrings, schedule and gather all parity computations 
        asynchronously. Returns a list of responses corresponding to each bitstring.
        A tqdm progress bar is displayed showing number of completed requests.
        """
        # Create a list of tasks
        tasks = [asyncio.create_task(self._multiply(x, y)) for x, y in tests]

        responses = []
        # Use as_completed to update the progress bar as each task finishes
        try:
            with tqdm(total=len(tasks)) as pbar:
                for coro in asyncio.as_completed(tasks, timeout=600):
                    try:
                        res = await coro
                    except JSONDecodeError as e:
                        print(f"Error: {e}")
                        continue
                    responses.append(res)
                    pbar.update(1)
        except asyncio.TimeoutError:
            print(f"Timeout reached. Returning {len(responses)} responses.")

        return responses
    

class Async4oMultiplicationCalculator:
    def __init__(self, api_key: str, n_max_tokens: int, model="o1-mini"):
        """
        Initialize the asynchronous OpenAI client and store the maximum tokens.
        """
        self.client = AsyncOpenAI(api_key=api_key, timeout=600)
        self.n_max_tokens = n_max_tokens
        self.model = model


    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    async def _multiply(self, x, y, **kwargs):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You will receive two numbers. You have to multiply them manually. "
                        f"Think step by step and be very careful. "
                        "Finish your response with the precise result of the multiplication."
                    )
                },
                {
                    "role": "user",
                    "content": f"{x} {y}"
                }
            ],
            stream=False,
        )
        # print(response.usage)
        return response.choices[0].message.content,  \
               x, y, response.usage.completion_tokens_details.reasoning_tokens

    async def run(self, tests):
        """
        Given a list of bitstrings, schedule and gather all parity computations 
        asynchronously. Returns a list of responses corresponding to each bitstring.
        A tqdm progress bar is displayed showing number of completed requests.
        """
        # Create a list of tasks
        tasks = [asyncio.create_task(self._multiply(x, y)) for x, y in tests]

        responses = []
        # Use as_completed to update the progress bar as each task finishes
        try:
            with tqdm(total=len(tasks)) as pbar:
                for coro in asyncio.as_completed(tasks, timeout=600):
                    try:
                        res = await coro
                    except JSONDecodeError as e:
                        print(f"Error: {e}")
                        continue
                    responses.append(res)
                    pbar.update(1)
        except asyncio.TimeoutError:
            print(f"Timeout reached. Returning {len(responses)} responses.")

        return responses


class Async4oParityCalculator:
    def __init__(self, api_key: str, model="o1-mini"):
        """
        Initialize the asynchronous OpenAI client and store the maximum tokens.
        """
        self.client = AsyncOpenAI(api_key=api_key, timeout=600)
        self.model = model


    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    async def _parity(self, bitstring, **kwargs):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You will receive a string. You have to manually calculate its parity. "
                        "Finish your response with 1 if the parity is odd, and 0 if the parity is even."
                    )
                },
                {
                    "role": "user",
                    "content": bitstring
                }
            ],
            stream=False,
        )
        # print(response.usage)
        return response.choices[0].message.content,  \
               bitstring, response.usage.completion_tokens_details.reasoning_tokens

    async def run(self, tests):
        """
        Given a list of bitstrings, schedule and gather all parity computations 
        asynchronously. Returns a list of responses corresponding to each bitstring.
        A tqdm progress bar is displayed showing number of completed requests.
        """
        # Create a list of tasks
        tasks = [asyncio.create_task(self._parity(bs)) for bs in tests]

        responses = []
        # Use as_completed to update the progress bar as each task finishes
        try:
            with tqdm(total=len(tasks)) as pbar:
                for coro in asyncio.as_completed(tasks, timeout=600):
                    try:
                        res = await coro
                    except JSONDecodeError as e:
                        print(f"Error: {e}")
                        continue
                    responses.append(res)
                    pbar.update(1)
        except asyncio.TimeoutError:
            print(f"Timeout reached. Returning {len(responses)} responses.")

        return responses
    


class AsyncMedianCalculator:
    def __init__(self, api_key: str):
        """
        Initialize the asynchronous OpenAI client and store the maximum tokens.
        """
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=600)

    async def _compute_median(self, string: str) -> str:
        """
        Send one asynchronous request to compute the parity of 'bitstring'.
        """
        response = await self.client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You will receive a sequence of numbers. You have to manually compute its median. "
                        "Finish your response with the value of the median of this sequence."
                    )
                },
                {
                    "role": "user",
                    "content": string
                }
            ],
            stream=False
        )
        return response.choices[0].message.reasoning_content + "\n-------\n" + response.choices[0].message.content,  \
               string, response.usage.completion_tokens

    async def run(self, strings):
        """
        Given a list of bitstrings, schedule and gather all parity computations 
        asynchronously. Returns a list of responses corresponding to each bitstring.
        A tqdm progress bar is displayed showing number of completed requests.
        """
        # Create a list of tasks
        tasks = [asyncio.create_task(self._compute_median(bs)) for bs in strings]

        responses = []
        # Use as_completed to update the progress bar as each task finishes
        try:
            with tqdm(total=len(tasks)) as pbar:
                for coro in asyncio.as_completed(tasks, timeout=600):
                    try:
                        res = await coro
                    except JSONDecodeError as e:
                        print(f"Error: {e}")
                        continue
                    responses.append(res)
                    pbar.update(1)
        except asyncio.TimeoutError:
            print(f"Timeout reached. Returning {len(responses)} responses.")

        return responses


class Asynco1MedianCalculator:
    def __init__(self, api_key: str, model="o1-mini"):
        """
        Initialize the asynchronous OpenAI client and store the maximum tokens.
        """
        self.client = AsyncOpenAI(api_key=api_key, timeout=600)
        self.model = model


    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    async def _median(self, string, **kwargs):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You will receive a sequence of numbers. You have to manually compute its median. "
                        "Finish your response with the value of the median of this sequence."
                    )
                },
                {
                    "role": "user",
                    "content": string
                }
            ],
            stream=False,
        )
        # print(response.usage)
        return response.choices[0].message.content,  \
               string, response.usage.completion_tokens_details.reasoning_tokens

    async def run(self, tests):
        """
        Given a list of bitstrings, schedule and gather all parity computations 
        asynchronously. Returns a list of responses corresponding to each bitstring.
        A tqdm progress bar is displayed showing number of completed requests.
        """
        # Create a list of tasks
        tasks = [asyncio.create_task(self._median(bs)) for bs in tests]

        responses = []
        # Use as_completed to update the progress bar as each task finishes
        try:
            with tqdm(total=len(tasks)) as pbar:
                for coro in asyncio.as_completed(tasks, timeout=600):
                    try:
                        res = await coro
                    except JSONDecodeError as e:
                        print(f"Error: {e}")
                        continue
                    responses.append(res)
                    pbar.update(1)
        except asyncio.TimeoutError:
            print(f"Timeout reached. Returning {len(responses)} responses.")

        return responses


async def one_multiplication_test(api_key, max_char, n_tests, n_max_tokens, random_seed, model="deepseek"):
    tests = []
    ans = {}
    np.random.seed(random_seed)
    for i in range(n_tests):
        x, y = np.random.randint(1, 10 ** max_char, 2).tolist()
        ans[(x, y)] = x * y
        tests.append((x, y))
    
    if model == "deepseek":
        calculator = AsyncMultiplicationCalculator(api_key=api_key, n_max_tokens=n_max_tokens)
    elif model == "o1-mini":
        calculator = Async4oMultiplicationCalculator(api_key=api_key, n_max_tokens=n_max_tokens, model=model)
    responses = await calculator.run(tests)

    results = []
    text_answers = []
    correct = 0
    for response, x, y, n_tokens in responses:
        if response:
            prediction = get_last_number(response)
            results.append({
                "x": x,
                "y": y,
                "prediction": prediction,
                "label": x * y,
                "n_tokens": n_tokens,
                "correct": prediction == x * y,
            })
            correct += results[-1]["correct"]
            text_answers.append(response)
    
    return results, correct / len(results), text_answers
    

async def one_parity_test(api_key, bitstring_size, n_tests, n_max_tokens, random_seed, model="deepseek"):
    bitstrings = []
    ans = {}
    np.random.seed(random_seed)
    for i in range(n_tests):
        bs = np.random.randint(0, 2, bitstring_size)
        parity = sum(bs) % 2
        bs_str = " ".join(map(str, bs))
        ans[bs_str] = parity
        bitstrings.append(bs_str)
    
    if model == "deepseek":
        calculator = AsyncParityCalculator(api_key=api_key, n_max_tokens=n_max_tokens)
    elif model == "o1-mini":
        calculator = Async4oParityCalculator(api_key=api_key, model=model)

    responses = await calculator.run(bitstrings)

    results = []
    text_answers = []
    correct = 0
    for response, bitstring, n_tokens in responses:
        if response:
            prediction = get_last_number(response)
            results.append({
                "bitstring": bitstring,
                "prediction": prediction,
                "label": int(ans[bitstring]),
                "n_tokens": n_tokens,
                "correct": bool(prediction == ans[bitstring]),
            })
            correct += results[-1]["correct"]
            text_answers.append(response)
    
    return results, correct / len(results), text_answers


def get_last_number(string):
    # Use regular expression to find all numbers in the string
    numbers = re.findall(r'\d+', string.replace(",", "").replace(".", "").replace("!", ""))
    # Return the last number if there are any, otherwise return None
    return int(numbers[-1]) if numbers else None


async def one_median_test(api_key, n_numbers, n_tests, random_seed, model="deepseek", max_number=1_000_000):
    strings = []
    ans = {}
    np.random.seed(random_seed)
    for i in range(n_tests):
        numbers = np.random.choice(max_number, n_numbers, replace=False).tolist()
        median = int(np.median(numbers))
        numbers_str = " ".join(map(str, numbers))
        ans[numbers_str] = median
        strings.append(numbers_str)
    
    if model == "deepseek":
        calculator = AsyncMedianCalculator(api_key=api_key)
    elif model == "o1-mini":
        calculator = Asynco1MedianCalculator(api_key=api_key, model=model)

    responses = await calculator.run(strings)

    results = []
    text_answers = []
    correct = 0
    for response, bitstring, n_tokens in responses:
        if response:
            prediction = get_last_number(response)
            results.append({
                "string": bitstring,
                "prediction": prediction,
                "label": int(ans[bitstring]),
                "n_tokens": n_tokens,
                "correct": bool(prediction == ans[bitstring]),
            })
            correct += results[-1]["correct"]
            text_answers.append(response)
    
    return results, correct / len(results), text_answers
