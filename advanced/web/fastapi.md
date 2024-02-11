# FastAPI Tutorial


## First Steps

The simplest FastAPI file could look like the following:

```py
    # main.py
    from fastapi import FastAPI

    app = FastAPI()


    @app.get("/")
    async def root():
        return {"message": "Hello World"}
```

This `app` is the same one referred by `uvicorn` in the command.

Run the live server:

```bash
    uvicorn main:app --reload
```


### OpenAPI

FastAPI generates a "schema" with all your API using the OpenAPI standard for defining APIs.

#### Schema

A _schema_ is a definition or description of something, just an abstract description.

### API schema

In this case, OpenAPI is a specification that dictates how to define a schema of your API.

This schema definition includes your API paths, the possible parameters they take, etc.

#### Data schema

The term schema might also refer to the shape of some data such as JSON content.

In that case, it would mean the JSON attributes and data types they have, etc.

#### OpenAPI and JSON Schema

OpenAPI defines an API schema for your API and that schema includes definitions (or "schemas") of the data sent and received by your API using JSON Schema, the standard for JSON data schemas.

#### Check the openapi.json

If you are curious about how the raw OpenAPI schema looks like, FastAPI automatically generates a JSON (schema) with the descriptions of all your API.

You can see it directly at: http://127.0.0.1:8000/openapi.json.

#### What is OpenAPI for¶

The OpenAPI schema is what powers the two interactive documentation systems included.

And there are dozens of alternatives, all based on OpenAPI. You could easily add any of those alternatives to your application built with FastAPI.

You could also use it to generate code automatically, for clients that communicate with your API. For example, frontend, mobile or IoT applications.


### Recap

Step 1: import FastAPI
Step 2: create a FastAPI instance
Step 3: create a path operation
Step 4: define the path operation function
Step 5: return the content

FastAPI is a Python class that provides all the functionality for your API.

Here the `app` variable will be an instance of the class FastAPI which will be the main point of interaction to create all your API.

_Path_ refers to the last part of the URL starting such as "/items/foo". 

A path is also commonly called an _endpoint_ or a _route_.

_Operation_ refers to one of the HTTP methods.

When building APIs, you normally use these specific HTTP methods to perform a specific action:

- POST: to create data.
- GET: to read data.
- PUT: to update data.
- DELETE: to delete data.

Define a path operation decorator

```py
    # import FastAPI
    from fastapi import FastAPI

    # create a FastAPI "instance"
    app = FastAPI()

    # create a path operation
    @app.get("/")
    async def root():
        """
        define the path operation function
        """
        # return the content
        return {"message": "Hello World"}
```

We can also use the other operations:

- @app.post()
- @app.put()
- @app.delete()


#### Step 4: define the path operation function

This is our path operation function:

- path: is /.
- operation: is get.
- function: is the function below the "decorator" (below @app.get("/")).

#### Step 5: return the content

We can return a dict, list, singular values as str, int, etc.

We can also return Pydantic models (you'll see more about that later).

There are many other objects and models that will be automatically converted to JSON (including ORMs, etc). 



## Path Parameters

With FastAPI, we get the folowing by using short, intuitive and standard Python type declarations:

- Editor support: error checks, autocompletion, etc.
- Data "parsing"
- Data validation
- API annotation and automatic documentation

We can declare path _parameters_ or _variables_ with the same syntax used by Python format strings:

```py
    from fastapi import FastAPI

    app = FastAPI()


    @app.get("/items/{item_id}")
    async def read_item(item_id):
        return {"item_id": item_id}
```

### Path parameters with types¶

We can declare the type of a path parameter in the function using standard Python type annotations:

```py
    from fastapi import FastAPI
    
    app = FastAPI()
    
    
    @app.get("/items/{item_id}")
    async def read_item(item_id: int):
        return {"item_id": item_id}
```

### Data conversion¶

If you run this example and open your browser at http://127.0.0.1:8000/items/3, you will see a response of:

```json
    {"item_id":3}
```

Notice that the value your function received (and returned) is 3, (a Python int) and not a string "3".

Thus, FastAPI gives us automatic request _parsing_.

### Data validation¶

But if we go to the browser at http://127.0.0.1:8000/items/foo, we see a nice HTTP error since the path parameter item_id had a value of "foo" which is not an int.

```json
{
    "detail": [
        {
            "loc": [
                "path",
                "item_id"
            ],
            "msg": "value is not a valid integer",
            "type": "type_error.integer"
        }
    ]
}
```

With the same Python type declaration, FastAPI gives us data validation.

Notice that the error also clearly states exactly the point where the validation did not pass which is incredibly helpful while developing and debugging code that interacts with your API.

### Documentation

When we open the browser to http://127.0.0.1:8000/docs, we see an automatic, interactive, API documentation:

FastAPI gives us automatic, interactive documentation (integrating Swagger UI).

### Pydantic¶

All the data validation is performed under the hood by Pydantic, so you get all the benefits from it. And you know you are in good hands.

You can use the same type declarations with str, float, bool and many other complex data types.

Several of these are explored in the next chapters of the tutorial.

### Order matters

Because path operations are evaluated in order, we need to make sure that the path for `/users/me` is declared before the one for `/users/{user_id}`.

Note that we cannot redefine a path operation such as `/users` since the first one will always be used since the path matches first.

```py
    from fastapi import FastAPI

    app = FastAPI()


    @app.get("/users/me")
    async def read_user_me():
        return {"user_id": "the current user"}


    @app.get("/users/{user_id}")
    async def read_user(user_id: str):
        return {"user_id": user_id}
```

Similarly, we cannot redefine a path operation. The first one will always be used since the path matches first.


### Predefined values

If we have a path operation that receives a path parameter but we want the possible valid path parameter values to be predefined, we can use a standard Python `Enum`.


```py
    from enum import Enum

    from fastapi import FastAPI

    # Create an Enum class
    class ModelName(str, Enum):
        alexnet = "alexnet"
        resnet = "resnet"
        lenet = "lenet"


    app = FastAPI()

    # Declare a path parameter
    @app.get("/models/{model_name}")
    async def get_model(model_name: ModelName):
        # Working with Python enumerations
        if model_name is ModelName.alexnet:
            return {"model_name": model_name, "message": "Deep Learning FTW!"}

        # Get the enumeration value
        if model_name.value == "lenet":
            return {"model_name": model_name, "message": "LeCNN all the images"}

        # Return enumeration members
        return {"model_name": model_name, "message": "Have some residuals"}
```

#### Create an Enum class

We import `Enum` and create a sub-class that inherits from `str` and from `Enum`.

By inheriting from `str` the API docs will be able to know that the values must be of type string and will be able to render correctly.

Then create class attributes with fixed values which will be the available valid values:

#### Declare a path parameter

Ceate a path parameter with a type annotation using the enum class you created (`ModelName`):


#### Working with Python enumerations

The value of the path parameter will be an enumeration member.

- Compare enumeration members
- Get the enumeration value
- Return enumeration members

In your client you will get a JSON response like:

```json
    {
      "model_name": "alexnet",
      "message": "Deep Learning FTW!"
    }
```


### Path parameters containing paths

Suppose we have a path operation with a path `/files/{file_path}`.

But you need file_path itself to contain a path such as `home/johndoe/myfile.txt`, so the URL for that file would be: `/files/home/johndoe/myfile.txt`.

- OpenAPI support
- Path convertor



## Query Parameters

When we declare other function parameters that are not part of the path parameters, they are automatically interpreted as _query_ parameters.

```py
    from fastapi import FastAPI

    app = FastAPI()

    fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


    @app.get("/items/")
    async def read_item(skip: int = 0, limit: int = 10):
        return fake_items_db[skip : skip + limit]
```

The query is the set of key-value pairs that go after the ? in a URL, separated by & characters.

For example, in the URL: http://127.0.0.1:8000/items/?skip=0&limit=10

the query parameters are:

- skip: with a value of 0
- limit: with a value of 10

When we declare them with Python types (such as int), they are converted to that type and validated against it.


### Defaults

Since query parameters are not a fixed part of a path, they can be optional and can have default values.

In the example above, they have default values of `skip=0` and `limit=10`.

### Optional parameters

We can also declare optional query parameters by setting their default to `None`.

In this case, the function parameter `q` will be optional and will be `None` by default.

```py
    from typing import Union
    from fastapi import FastAPI

    app = FastAPI()


    @app.get("/items/{item_id}")
    async def read_item(item_id: str, q: Union[str, None] = None):
        if q:
            return {"item_id": item_id, "q": q}
        return {"item_id": item_id}
```

**NOTE:** FastAPI is smart enough to notice that the path parameter item_id is a path parameter and q is not, so, it is a query parameter.

### Query parameter type conversion

We can also declare bool types, and they will be converted:

```py
    from typing import Union
    from fastapi import FastAPI

    app = FastAPI()


    @app.get("/items/{item_id}")
    async def read_item(item_id: str, q: Union[str, None] = None, short: bool = False):
        item = {"item_id": item_id}
        if q:
            item.update({"q": q})
        if not short:
            item.update(
                {"description": "This is an amazing item that has a long description"}
            )
        return item
```

### Multiple path and query parameters

We can declare multiple path parameters and query parameters at the same time, FastAPI knows which is which.

We do not have to declare them in any specific order since they will be detected by name:

```py
    from typing import Union
    from fastapi import FastAPI

    app = FastAPI()


    @app.get("/users/{user_id}/items/{item_id}")
    async def read_user_item(
        user_id: int, item_id: str, q: Union[str, None] = None, short: bool = False
    ):
        item = {"item_id": item_id, "owner_id": user_id}
        if q:
            item.update({"q": q})
        if not short:
            item.update(
                {"description": "This is an amazing item that has a long description"}
            )
        return item
```

### Required query parameters

When we declare a default value for non-path parameters (for now, we have only seen query parameters), it is not required.

If we do not want to add a specific value but just make it optional, set the default as `None`.

When we want to make a query parameter required, we do not declare any default value:

```py
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/items/{item_id}")
    async def read_user_item(item_id: str, needy: str):
    """
    needy is a required query parameter of type str
    """
        item = {"item_id": item_id, "needy": needy}
        return item
```


## Request Body

When we need to send data from a client (such as a browser) to the API, we send it as a _request body_.

A request body is data sent by the client to your API. 

A response body is the data your API sends to the client.

The API almost always has to send a response body, but clients do not necessarily need to send request bodies all the time.

To declare a request body, you use Pydantic models with all their power and benefits.

**NOTE:** To send data, we should use: POST (the more common), PUT, DELETE, or PATCH.

```py
    from typing import Union
    from fastapi import FastAPI
    from pydantic import BaseModel

    # create the data model
    class Item(BaseModel):
        name: str
        description: Union[str, None] = None
        price: float
        tax: Union[float, None] = None

    app = FastAPI()

    @app.post("/items/")
    # declare it as a parameter
    async def create_item(item: Item):
        return item
```

### Import Pydantic BaseModel

First, we need to import `BaseModel` from `pydantic`:

### Create the data model

Then yweou declare your data model as a class that inherits from `BaseModel`.

We can use standard Python types for all the attributes:

When a model attribute has a default value, it is not required. Otherwise, it is required. 

Use `None` to make it just optional.

For example, this model above declares a JSON "object" (or Python dict) like:

```json
    {
        "name": "Foo",
        "description": "An optional description",
        "price": 45.2,
        "tax": 3.5
    }
```

### Declare it as a parameter

To add the data model to the path operation, declare it the same way we declared path and query parameters:


### Results¶

With just that Python type declaration, FastAPI will:

- Read the body of the request as JSON.
- Convert the corresponding types (if needed).
- Validate the data.

  If the data is invalid, it will return a nice and clear error, indicating exactly where and what was the incorrect data.

- Give us the received data in the parameter item.

  Since we declared it in the function to be of type Item, we will also have editor support (completion, etc) for all of the attributes and their types.

- Generate JSON Schema definitions for your model, we can also use them anywhere else we like.

- The schemas will be part of the generated OpenAPI schema,\ and used by the automatic documentation UIs.

### Automatic docs¶

The JSON Schemas of your models will be part of the OpenAPI generated schema and will be shown in the interactive API docs:

And will be also used in the API docs inside each path operation that needs them:

### Use the model

Inside of the function, we can access all the attributes of the model object directly:

```py
    from typing import Union
    from fastapi import FastAPI
    from pydantic import BaseModel

    class Item(BaseModel):
        name: str
        description: Union[str, None] = None
        price: float
        tax: Union[float, None] = None

    app = FastAPI()

    @app.post("/items/")
    async def create_item(item: Item):
        item_dict = item.dict()
        if item.tax:
            # use the model
            price_with_tax = item.price + item.tax
            item_dict.update({"price_with_tax": price_with_tax})
        return item_dict
```

### Request body + path parameters

You can declare path parameters and request body at the same time.

FastAPI will recognize that the function parameters that match path parameters should be taken from the path, and that function parameters that are declared to be Pydantic models should be taken from the request body.

### Request body + path + query parameters

The function parameters will be recognized as follows:

- If the parameter is declared in the path, it will be used as a path parameter.

- If the parameter is a singular type (such as int, float, str, bool), it will be interpreted as a query parameter.

- If the parameter is declared to be of the type of a Pydantic model, it will be interpreted as a request body.

### Request body + path + query parameters¶

We can also declare body, path, and query parameters (all at the same time).

FastAPI will recognize each of them and take the data from the correct place.

```py
    from typing import Union
    from fastapi import FastAPI
    from pydantic import BaseModel

    class Item(BaseModel):
        name: str
        description: Union[str, None] = None
        price: float
        tax: Union[float, None] = None

    app = FastAPI()

    @app.put("/request/items/{item_id}")
    async def create_item_query(item_id: int, item: Item, q: Union[str, None] = None):
        result = {"item_id": item_id, **item.dict()}

        if q:
            result.update({"q": q})

        return result
```



## Query Parameters and String Validations

FastAPI allows you to declare additional information and validation for your parameters.

We can declare additional validations and metadata for parameters.

Generic validations and metadata:

- alias
- title
- description
- deprecated

Validations specific for strings:

- min_length
- max_length
- regex

In the examples so far, we saw how to declare validations for `str` values.

The next chapters will show how to declare validations for other types such as numbers.

```py
from typing import Union
from fastapi import FastAPI

app = FastAPI()

@app.get("/params/items/")
async def read_items(q: Union[str, None] = None):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results
```

### Additional validation

We can enforce that even though `q` is optional (when it is provided) its length does not exceed 50 characters.

```py
    from typing import Union
    from fastapi import FastAPI, Query  # import Query

    app = FastAPI()

    @app.get("/params/items/")
    async def read_items(q: Union[str, None] = Query(default=None, max_length=50)):
        results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
        if q:
            results.update({"q": q})
        return results
```

### Use Query as the default value

We can now use Query as the default value of the parameter, setting the parameter `max_length` to 50:

`q: Union[str, None] = Query(default=None)` makes the parameter optional the same as `q: Union[str, None] = None `but we can also pass more parameters to `Query`.

### Add more validations

We can also add a parameter `min_length`:

### Add regular expressions

We can define a regular expression that the parameter should match:

### Default values

The same way that we can pass `None` as the value for the default parameter, we can pass other values.

We can declare the `q` query parameter to have a `min_length` of 3 and a default value of "fixedquery":

```py
    @app.get("/params/default/")
    async def read_items(q: str = Query(default="fixedquery", min_length=3)):
        """
        q query parameter has min_length of 3 with default value of "fixedquery".
        Having a default value also makes the parameter optional.
        """
        results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
        if q:
            results.update({"q": q})
        return results
```

### Make it required

When we do not need to declare more validations or metadata, we can make the `q` query parameter required by not declaring a default value, 

```py
    from fastapi import FastAPI, Query

    app = FastAPI()

    @app.get("/items/")
    async def read_items(q: str = Query(min_length=3)):
        results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
        if q:
            results.update({"q": q})
        return results
```

#### Required with Ellipsis (...)

We can also declare that a value is required using the literal value ...:

```py
    @app.get("/params/ellipsis/")
    async def read_items(q: str = Query(default=..., min_length=3)):
        results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
        if q:
            results.update({"q": q})
        return results
```

#### Required with None

We can declare that a parameter can accept None, but that it is still required which will force clients to send a value, even if the value is `None`.

We declare that `None` is a valid type but still use `default=...`:

```py
    @app.get("/params/none/")
    async def read_items(q: Union[str, None] = Query(default=..., min_length=3)):
        results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
        if q:
            results.update({"q": q})
        return results
```

#### Use Pydantic Required instead of Ellipsis (...)

If you feel uncomfortable using `...`, you can also import and use `Required` from Pydantic:

```py
    from fastapi import FastAPI, Query
    from pydantic import Required

    app = FastAPI()

    @app.get("/items/")
    async def read_items(q: str = Query(default=Required, min_length=3)):
        results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
        if q:
            results.update({"q": q})
        return results
```

**NOTE:** When something is required, we can usually omit the `default` parameter, so you normally do not have to use `...` or `Required`.


### Query parameter list / multiple values

When we define a query parameter explicitly with `Query`, we can also declare it to receive a list of values (multiple values).

We can declare a query parameter `q` that can appear multiple times in the URL:

```py
    from typing import List, Union
    from fastapi import FastAPI, Query

    app = FastAPI()

    @app.get("/params/multiple/")
    async def read_items(q: Union[List[str], None] = Query(default=None)):
        query_items = {"q": q}
        return query_items
```

**NOTE:** To declare a query parameter with a type of `list`, we need to explicitly use `Query` ore it would be interpreted as a request body.


### Query parameter list / multiple values with defaults

We can also define a default `list` of values if none are provided:

However, FastAPI will not check the contents of the list.

```py
    from typing import List
    from fastapi import FastAPI, Query

    app = FastAPI()

    @app.get("/params/multiple/defaults/")
    async def read_items(q: List[str] = Query(default=["foo", "bar"])):
        query_items = {"q": q}
        return query_items
```

### Declare more metadata

We can add more information about the parameter that will be included in the generated OpenAPI and used by the documentation user interfaces and external tools.

```py
    from typing import Union
    from fastapi import FastAPI, Query

    app = FastAPI()

    @app.get("/params/metadata/")
    async def read_items(
        q: Union[str, None] = Query(
            default=None,
            title="Query string",
            description="Query string for the items to search in the database that have a good match",
            min_length=3,
        )
    ):
        results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
        if q:
            results.update({"q": q})
        return results
```

### Alias parameters¶

Suppose we want the parameter to be item-query.

http://127.0.0.1:8000/items/?item-query=foobaritems

But `item-query` is not a valid Python variable name.

```py
    from typing import Union
    from fastapi import FastAPI, Query

    app = FastAPI()

    # http://127.0.0.1:8000/params/alias/?item-query=foobaritems
    @app.get("/params/alias/")
    async def read_items(q: Union[str, None] = Query(default=None, alias="item-query")):
        results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
        if q:
            results.update({"q": q})
        return results
```

### Deprecating parameters

Suppose we do not like this parameter anymore.

We have to leave it there a while because there are clients using it, but we want the docs to clearly show it as deprecated.

Then pass the parameter `deprecated=True` to `Query`:

```py
    from typing import Union
    from fastapi import FastAPI, Query

    app = FastAPI()

    @app.get("/params/deprecated/")
    async def read_items(
        q: Union[str, None] = Query(
            default=None,
            alias="item-query",
            title="Query string",
            description="Query string for the items to search in the database that have a good match",
            min_length=3,
            max_length=50,
            regex="^fixedquery$",
            deprecated=True,
        )
    ):
        results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
        if q:
            results.update({"q": q})
        return results
```

### Exclude from OpenAPI

To exclude a query parameter from the generated OpenAPI schema (and the automatic documentation systems), set the parameter `include_in_schema` of `Query` to `False`.

```py
    from typing import Union
    from fastapi import FastAPI, Query

    app = FastAPI()

    @app.get("/items/")
    async def read_items(
        hidden_query: Union[str, None] = Query(default=None, include_in_schema=False)
    ):
        if hidden_query:
            return {"hidden_query": hidden_query}
        else:
            return {"hidden_query": "Not found"}
```



## Path Parameters and Numeric Validations

In the same way that we can declare more validations and metadata for query parameters with `Query`, we can declare the same type of validations and metadata for path parameters with `Path`.

With `Query`, `Path` (and others), we can declare metadata and string validations in the same ways as with "Query Parameters and String Validations".

We can also declare numeric validations:

- gt: greater than
- ge: greater than or equal
- lt: less than
- le: less than or equal

**NOTE:** Query, Path, and other classes you will see later are subclasses of a common Param class. All of them share the same parameters for additional validation and metadata you have seen.

### Technical Details

When we import `Query`, `Path` and others from `fastapi`, they are actually **functions** return instances of classes of the same name.

So, we import `Query` which is a function and when we call the function, it returns an instance of a class also named `Query`.

These functions are there (instead of just using the classes directly) so that the editor does not mark errors about their types.

This way we can use a normal editor and coding tools without having to add custom configurations to disregard those errors.

### Import Path

First, import `Path` from `fastapi`.

### Declare metadata

We can declare all the same parameters as for `Query`.

We can declare a `title` metadata value for the `path` parameter `item_id`:

```py
    from typing import Union
    from fastapi import FastAPI, Path, Query

    app = FastAPI()

    @app.get("/items/{item_id}")
    async def read_items(
        item_id: int = Path(title="The ID of the item to get"),
        q: Union[str, None] = Query(default=None, alias="item-query"),
    ):
        results = {"item_id": item_id}
        if q:
            results.update({"q": q})
        return results
```

### Order the parameters as you need

Suppose we want to declare the query parameter `q` as a required `str`.

We do not need to declare anything else for that parameter, so we do not really need to use `Query`.

But we still need to use `Path` for the `item_id` path parameter.

Python will complain if we define a value with a "default" before a value that does not have a "default".

But we can re-order them and have the value without a default (the query parameter `q`) first.

FastAPI will detect the parameters by their names, types, and default declarations (Query, Path, etc); it does not care about the order.

```py
    @app.get("/items/{item_id}")
    async def read_items(q: str, item_id: int = Path(title="The ID of the item to get")):
        results = {"item_id": item_id}
        if q:
            results.update({"q": q})
        return results
```

### Order the parameters as you need (tricks)

If we want to declare the `q` query parameter without a `Query` nor any default value and the path parameter `item_id` using `Path` and we have them in a different order, Python has a special syntax for that.

Pass `*` as the first parameter of the function.

Python will not do anything with that `*` but it will know that all the following parameters should be called as keyword arguments (key-value pairs), also known as kwargs. Even if they don't have a default value.

```py
    @app.get("/items/{item_id}")
    async def read_items(*, item_id: int = Path(title="The ID of the item to get"), q: str):
        results = {"item_id": item_id}
        if q:
            results.update({"q": q})
        return results
```

### Number validations: greater than or equal

With `Query` and `Path` (and others) we can declare number constraints.

Here, with `ge=1`, `item_id` will need to be an integer number "greater than or equal" to 1.

```py
    @app.get("/items/{item_id}")
    async def read_items(
        *, item_id: int = Path(title="The ID of the item to get", ge=1), q: str
    ):
        results = {"item_id": item_id}
        if q:
            results.update({"q": q})
        return results
```

### Number validations: greater than and less than or equal

The same applies for:

- gt: greater than
- le: less than or equal

```py
    @app.get("/items/{item_id}")
    async def read_items(
        *,
        item_id: int = Path(title="The ID of the item to get", gt=0, le=1000),
        q: str,
    ):
        results = {"item_id": item_id}
        if q:
            results.update({"q": q})
        return results
```

### Number validations: floats, greater than and less than¶

Number validations also work for `float` values.

Here is where it becomes important to be able to declare `gt` and not just `ge` since we can require that a value must be greater than 0, even if it is less than 1.

Therfore, 0.5 would be a valid value, but 0.0 or 0 would not.

And the same for `lt`.

```py
    @app.get("/items/{item_id}")
    async def read_items(
        *,
        item_id: int = Path(title="The ID of the item to get", ge=0, le=1000),
        q: str,
        size: float = Query(gt=0, lt=10.5)
    ):
        results = {"item_id": item_id}
        if q:
            results.update({"q": q})
        return results
```



## Body - Multiple Parameters¶

Now that we have seen how to use `Path` and `Query`, we can discuss more advanced uses of request body declarations.

### Mix Path, Query and body parameters¶

We can mix `Path`, `Query`, and request body parameter declarations freely and FastAPI will know what to do.

We can also declare body parameters as optional by setting the default to `None`:



## Extra Data Types¶

We have been using common data types:

- int
- float
- str
- bool

But you can also use more complex data types.

### Other data types

Here are some of the additional data types we can use:

- UUID
- datetime.datetime
- datetime.time
- datetime.timedelta
- frozenset
- bytes
- Decimal

You can check all the valid pydantic data types: [Pydantic data types](https://pydantic-docs.helpmanual.io/usage/types).

### Example

Here is an example path operation with parameters using some of the above types:



## Extra Models

Continuing with the previous example, it will be common to have more than one related model.

This is especially the case for user models because:

- The input model needs to be able to have a password.
- The output model should not have a password.
- The database model would probably need to have a hashed password.

**NOTE:** Never store user plaintext passwords. Always store a "secure hash" that you can then verify which is discussed in the [security chapters](https://fastapi.tiangolo.com/tutorial/security/simple-oauth2/#password-hashing).


## Multiple models

Here is a general idea of how the models could look like with their password fields and the places where they are used:


## Response Status Code

The same way we can specify a response model, we can also declare the HTTP status code used for the response with the parameter `status_code` in any of the path operations:

```py
    from fastapi import FastAPI
    
    app = FastAPI()
    
    @app.post("/items/", status_code=201)
    async def create_item(name: str):
        return {"name": name}
```

The `status_code` parameter receives a number with the HTTP status code.

**NOTE:** The `status_code` is a parameter of the decorator method (get, post, etc), not the path operation function which is true for all the parameters and body.


### About HTTP status codes

In HTTP, we send a numeric status code of 3 digits as part of the response.

These status codes have a name associated to recognize them, but the important part is the number:

- 100 and above are for "Information". You rarely use them directly. Responses with these status codes cannot have a body.

- 200 and above are for "Successful" responses. These are the ones you would use the most.

  200 is the default status code, which means everything was "OK".

  Another example would be 201, "Created". It is commonly used after creating a new record in the database.

  A special case is 204, "No Content". This response is used when there is no content to return to the client, and so the response must not have a body.

- 300 and above are for "Redirection". Responses with these status codes may or may not have a body, except for 304, "Not Modified", which must not have one.

- 400 and above are for "Client error" responses. These are the second type you would probably use the most.
  
  An example is 404, for a "Not Found" response.

  For generic errors from the client, you can just use 400.

- 500 and above are for server errors. You almost never use them directly. When something goes wrong at some part in your application code, or server, it will automatically return one of these status codes.


### Shortcut to remember the names

We do not have to memorize what each of these codes mean since we can use the convenience variables from `fastapi.status`.

```py
    from fastapi import FastAPI, status

    app = FastAPI()

    @app.post("/items/", status_code=status.HTTP_201_CREATED)
    async def create_item(name: str):
        return {"name": name}
```

NOTE: FastAPI provides the same starlette.status as fastapi.status just as a convenience, but it comes directly from Starlette.

### Changing the default

Later, in the Advanced User Guide, you will see how to return a different status code than the default you are declaring here.


## Form Data

When we need to receive form fields instead of JSON, we can use `Form`.

We use `Form` to declare form data input parameters.

```py
    from fastapi import FastAPI, Form

    app = FastAPI()

    @app.post("/login/")
    async def login(username: str = Form(), password: str = Form()):
        return {"username": username}
```


## Handling Errors

There are many situations where we need to notify an error to a client that is using the API.

This client could be a browser with a frontend, a code from someone else, an IoT device, etc.

We may need to tell the client that:

- The client does not have enough privileges for that operation.
- The client does not have access to that resource.
- The item the client was trying to access does not exist.

In these cases, we would normally return an HTTP status code in the range of 400 (from 400 to 499).

This is similar to the 200 HTTP status codes (from 200 to 299). The "200" status codes mean that somehow there was a "success" in the request.

The status codes in the 400 range mean that there was an error from the client.

### Use HTTPException

To return HTTP responses with errors to the client we use `HTTPException`.

```py
    from fastapi import FastAPI, HTTPException

    app = FastAPI()

    items = {"foo": "The Foo Wrestlers"}

    @app.get("/items/{item_id}")
    async def read_item(item_id: str):
        if item_id not in items:
            raise HTTPException(status_code=404, detail="Item not found")
        return {"item": items[item_id]}
```


## Dependencies - First Steps

FastAPI has a very powerful but intuitive **Dependency Injection** system that is designed to be very simple to use and to make it very easy for any developer to integrate other components with FastAPI.

### What is Dependency Injection

Dependency Injection means that there is a way for  code (the path operation functions) to declare things that it requires to work and use: _dependencies_.

Then the system (FastAPI) will take care of doing whatever is needed to provide your code with those needed dependencies (inject the dependencies).

Dependency Injection is very useful when you need to:

- Have shared logic (the same code logic again and again).
- Share database connections.
- Enforce security, authentication, role requirements, etc.
- And more.

All these while minimizing code repetition.


## Middleware

We can add middleware to FastAPI applications.

A middleware is a function that works with every **request** before it is processed by any specific _path operation_, and also with every **response** before returning it.

- It takes each request that comes to your application.
- It can then do something to that request or run any needed code.
- Then it passes the request to be processed by the rest of the application (by some path operation).
- It then takes the response generated by the application (by some path operation).
- It can do something to that response or run any needed code.
- Then it returns the response.


## CORS (Cross-Origin Resource Sharing)¶

**Cross-Origin Resource Sharing (CORS)** refers situations when a frontend running in a browser has JavaScript code that communicates with a backend and the backend is in a different _origin_ than the frontend.

### Origin

An _origin_ is the combination of protocol (http, https), domain (myapp.com, localhost, localhost.tiangolo.com), and port (80, 443, 8080).

Therfore, all these are different origins:

- http://localhost
- https://localhost
- http://localhost:8080

Even if they are all in localhost, they use different protocols or ports, so they are different origins.


## SQL (Relational) Databases¶

FastAPI does not require you to use a SQL (relational) database, but you can use any relational database that you want.

Here we will see an example using SQLAlchemy.

You can easily adapt it to any database supported by SQLAlchemy:

- PostgreSQL
- MySQL
- SQLite
- Oracle
- Microsoft SQL Server

In this example, we will use SQLite because it uses a single file and Python has integrated support. 

Later, will probably want to use a database server such as PostgreSQL.

### ORMs¶

FastAPI works with any database and any style of library to talk to the database.

A common pattern is to use an an "object-relational mapping (ORM) library.

An ORM has tools to convert (map) between objects in code and database tables (relations).

With an ORM, we normally create a class that represents a table in a SQL database in which each attribute of the class represents a column with a name and a type.

For example a class `Pet` could represent a SQL table `pets`.

And each instance object of that class represents a row in the database.

For example an object `orion_cat` (an instance of `Pet`) could have an attribute `orion_cat.type` for the column type and the value of that attribute could be "cat".

These ORMs also have tools to make the connections or relations between tables or entities.

This way, we can also have an attribute `orion_cat.owner` and the owner would contain the data for this pet's owner, taken from the table `owners`.

So, `orion_cat.owner.name` could be the name (from the `name` column in the `owners` table) of this pet's owner with a value such as "Arquilian".

And the ORM will do all the work to get the information from the corresponding table `owners` when we try to access it from the pet object.

Common ORMs are: Django-ORM (part of the Django framework), SQLAlchemy ORM (part of SQLAlchemy, independent of framework), and Peewee (independent of framework).

Here we will see how to work with SQLAlchemy ORM.


## Bigger Applications - Multiple Files

If we are building an application or a web API, it is rarely the case that we can put everything on a single file.

FastAPI provides a convenience tool to structure your application while keeping all the flexibility.

NOTE: If you come from Flask, this would be the equivalent of Flask's Blueprints.

### An example file structure

Suppose we have the following file structure:


## Background Tasks¶

We can define background tasks to be run _after_ returning a response.

This is useful for operations that need to happen after a request, but that the client does not really have to be waiting for the operation to complete before receiving the response.

Here are some examples:

- Email notifications sent after performing an action:

  Since connecting to an email server and sending an email tends to be "slow" (several seconds), we can return the response right away and send the email notification in the background.

- Processing data:

  If we receive a file that must go through a slow process, we can return a response of "Accepted" (HTTP 202) and process it in the background.


## Metadata and Docs URLs¶

We can customize several metadata configurations in your FastAPI application.

### Metadata for API

We can set the following fields that are used in the OpenAPI specification and the automatic API docs UIs:


## Static Files

We can serve static files automatically from a directory using `StaticFiles`.

```py
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    app = FastAPI()

    # mount StaticFiles instance to a specific path
    app.mount("/static", StaticFiles(directory="static"), name="static")
```


## Testing¶

Thanks to Starlette, testing FastAPI applications is easy and enjoyable.

Testing is based on `Requests`, so it is very familiar and intuitive.

We can also use `pytest` directly with FastAPI.


## Debugging

We can connect the debugger in the editor such as Visual Studio Code or PyCharm.


## Concurrency and async / await

Details about the `async def` syntax for _path operation functions_ and some background about asynchronous code, concurrency, and parallelism.

### TL;DR

If we are using third party libraries that are called with await:

```py
    results = await some_library()
```

Then, declare your _path operation functions_ with `async def`:

```py
    @app.get('/')
    async def read_results():
        results = await some_library()
        return results
```

**NOTE:** We can only use `await` inside of functions created with `async def`.

If we are using a third party library that communicates with something (a database, an API, the file system, etc.) and does ot have support `await` (which is currently the case for most database libraries) then declare the _path operation functions_ with just `def`:

```py
    @app.get('/')
    def results():
        results = some_library()
        return results
```

If the application does not have to communicate with anything else and wait for it to respond, use `async def`.

If you are not sure then use `def`.

**NOTE:** We can mix `def` and `async def` in path operation functions as much as we need and define each one using the best option for each use case. FastAPI will do the right thing with them.

In any of the cases above, FastAPI will still work asynchronously and be extremely fast.

But by following the steps above, it will be able to do some performance optimizations.


## Deployment - Intro¶

Deploying a FastAPI application is relatively easy.

### What Does Deployment Mean

To deploy an application means to perform the necessary steps to make it **available to the users**.

For a web API, this normally involves putting it on a remote machine (with a server program that provides good performance, stability, etc) so that users can access the application efficiently and without interruptions or problems.

This is in contrast to the development stages where we are constantly changing the code, breaking it and fixing it, stopping and restarting the development server, etc.

### Deployment Strategies

There are several ways to deploy depending on the specific use case and the tools that we use.

We could deploy a server using a combination of tools, we could use a cloud service that does part of the work for us, or other possible options.

Here we discuss some of the main concepts we should keep in mind when deploying a FastAPI application (although most of it applies to any other type of web application).




## References

[FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial)