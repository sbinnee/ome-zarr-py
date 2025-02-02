import pathlib

import numpy as np
import pytest
import zarr

from ome_zarr.format import FormatV01, FormatV02, FormatV03
from ome_zarr.io import parse_url
from ome_zarr.reader import Multiscales, Reader
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image


class TestWriter:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = pathlib.Path(tmpdir.mkdir("data"))
        self.store = parse_url(self.path, mode="w").store
        self.root = zarr.group(store=self.store)
        self.group = self.root.create_group("test")

    def create_data(self, shape, dtype=np.uint8, mean_val=10):
        rng = np.random.default_rng(0)
        return rng.poisson(mean_val, size=shape).astype(dtype)

    @pytest.fixture(
        params=(
            (1, 2, 1, 256, 256),
            (3, 512, 512),
            (256, 256),
        ),
        ids=["5D", "3D", "2D"],
    )
    def shape(self, request):
        return request.param

    @pytest.fixture(params=[True, False], ids=["scale", "noop"])
    def scaler(self, request):
        if request.param:
            return Scaler()
        else:
            return None

    @pytest.mark.parametrize(
        "format_version",
        (
            pytest.param(
                FormatV01,
                id="V01",
                marks=pytest.mark.xfail(reason="issues with dimension_separator"),
            ),
            pytest.param(FormatV02, id="V02"),
            pytest.param(FormatV03, id="V03"),
        ),
    )
    def test_writer(self, shape, scaler, format_version):

        data = self.create_data(shape)
        version = format_version()
        axes = "tczyx"[-len(shape) :]
        write_image(
            image=data,
            group=self.group,
            chunks=(128, 128),
            scaler=scaler,
            fmt=version,
            axes=axes,
        )

        # Verify
        reader = Reader(parse_url(f"{self.path}/test"))
        node = list(reader())[0]
        assert Multiscales.matches(node.zarr)
        if version.version not in ("0.1", "0.2"):
            # v0.1 and v0.2 MUST be 5D
            assert node.data[0].shape == shape
        else:
            assert node.data[0].ndim == 5
        assert np.allclose(data, node.data[0][...].compute())
