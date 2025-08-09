# LLM Prompt Engineering

Here some methods and techniques to improve prompt engineering for LLM applications.

## Ensure output consistency

### Markup tags

We can use a technique in which LLM answers are enclosed in markup tags to ensure output consistency.

```py
prompt = f"""
Classify the text into "Cat" or "Dog"

Provide your response in <answer> </answer> tags
"""
```

Now we can easily parse out the response using the following code:

```py
def _parse_response(response: str):
    return response.split("<answer>")[1].split("</answer>")[0]

```

The reason that markup tags works so well is that this is how the model is trained to behave.

When OpenAI, Google, and others train the models, they use markup tags. Therefore, the models are very effective at utilizing these tags and will usually adhere to the expected response format.

We can also make use of markup tags elsewhere in our prompts.

Here we provide few shot examples to a model, we can do the following:

```py
prompt = f"""
Classify the text into "Cat" or "Dog"

Provide your response in <answer> </answer> tags

<example>
This is an image showing a cat -> <answer>Cat</answer>
</example>
<example>
This is an image showing a dog -> <answer>Dog</answer>
</example>
"""
```

### Output validation

We can use Pydantic to validate the output of an LLM. We can define types and validate the output of the model.

### Handling errors

Errors are inevitable when dealing with LLMs.

Therefore, it is important to use the following techniques to handle errors:

- Retry mechanism with exponential backoff
- Increase the temperature
- Have backup LLMs

## References

[1]: [How to Ensure Reliability in LLM Applications](https://towardsdatascience.com/how-to-ensure-reliability-in-llm-applications/)
