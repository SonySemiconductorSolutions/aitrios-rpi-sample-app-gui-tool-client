<h1 align='center'><b>Unify</b></h1>
<p align='center'>
The guitool client is created with an SDK supporting application development on the AI Camera.
</p>

## 1. Development environment setup.
```
make setup
```

> Unit-tests are included in the corresponding `/tests` folder and can be exectued by running:
> `make test`  
> This also generates a corresponding coverage report.
>
> Linting corrections and checks (black, isort & flake8) by running:
> `make lint`

## 2. Building the SDK wheel.
One can create a wheel file of the Unify SDK by running:
```
make build
```
The generated wheel file located in the `/dist` folder can be used to install the sdk in you project environment. (e.g. `pip install unify-<version>-py3-none-any.whl`)

## Basic example

As a basic example let's run MobilenetV2 using the the unify SDK.
1. First make sure you have downloaded the `imx500_network_mobilenet_v2.rpk` file from the [Model Zoo](https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_mobilenet_v2.rpk)
2. Create a new file `example.py` with the following content.

```python
from unify.devices import AiCamera
from unify.models import Model, MODEL_TYPE, COLOR_FORMAT
from unify.models.post_processors import pp_cls

class MobilenetV2(Model):
    def __init__(self):
        super().__init__(
            model_file="./imx500_network_mobilenet_v2.rpk",
            model_type=MODEL_TYPE.PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=True
        )

    def post_process(self, output_tensors):
        return pp_cls(output_tensors)

device = AiCamera()
model = MobilenetV2()
device.deploy(model)

with device as stream:
    for frame in stream:
        print(frame.detections)
        frame.display()
```

3. Run the example, and make sure unify is installed in your environment.

```
. .venv/bin/activate && python example.py
```

Note that the unify API allows you to combine any network.rpk with your own custom post_processing function. Or choose any post processor function form the provided list inside `unify > models > post_processors`

## Releases

Release tags must be of the format "\d+\.\d+\.\d+" example "1.0.4".

## License

[LICENSE](./LICENSE)

## Notice

### Security

Please read the Site Policy of GitHub and understand the usage conditions.