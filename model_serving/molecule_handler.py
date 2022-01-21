import io
import os
import json
import base64
import time
from abc import ABC

import torch
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from ts.torch_handler.base_handler import BaseHandler


class MoleculeHandler(BaseHandler, ABC):
    """
    Handler class for molecules
    """
    def initialize(self, context):
        super().initialize(context)


    def preprocess(self, data):
        """The preprocess function converts the input data to a float tensor
        Args:
            data (List): Input data from the request is in the form of a Tensor
        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """

        molecules = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)


#        z = torch.tensor(data["z"], device=self.device)
#        pos = torch.tensor(data["pos"], device=self.device)
#        batch = torch.tensor(data["batch"], device=self.device)

#        data = tuple(z, pos, batch)

        return data


    def inference(self, data, *args, **kwargs):
        """
        The Inference Function is used to make a prediction call on the given input request.
        The user needs to override the inference function to customize it.
        Args:
            data (Torch Tensor): A Torch Tensor is passed to make the Inference Request.
            The shape should match the model input shape.
        Returns:
            Torch Tensor : The Predicted Torch Tensor is returned in this function.
        """
        # marshalled_data = data.to(self.device)
        for _ in range(20):
            print('##########################################################')
        data = json.loads(data[0]['data'].decode('ascii'))
        print(data)
        for _ in range(20):
            print('##########################################################')

        z = torch.tensor(data['z']).to(self.device)
        pos = torch.tensor(data['pos']).to(self.device)
        batch = torch.tensor(data['batch']).to(self.device)

        with torch.no_grad():
            results = self.model(z, pos, batch, *args, **kwargs)
        return results


    def handle(self, data, context):
        """Entry point for default handler. It takes the data from the input request and returns
           the predicted outcome for the input.
        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artefacts parameters.
        Returns:
            list : Returns a list of dictionary with the predicted response.
        """

        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        start_time = time.time()

        self.context = context
        metrics = self.context.metrics

        is_profiler_enabled = os.environ.get("ENABLE_TORCH_PROFILER", None)
        if is_profiler_enabled:
            output, _ = self._infer_with_profiler(data=data)
        else:
            data_preprocess = self.preprocess(data)

            if not self._is_explain():
                output = self.inference(data_preprocess)
                output = self.postprocess(output)
            else:
                output = self.explain_handle(data_preprocess, data)

        stop_time = time.time()
        metrics.add_time('HandlerTime', round(
            (stop_time - start_time) * 1000, 2), None, 'ms')
        return output


    def postprocess(self, data):
        """
        The post process function makes use of the output from the inference and converts into a
        Torchserve supported response output.
        Args:
            data (Torch Tensor): The torch tensor received from the prediction output of the model.
        Returns:
            List: The post process function returns a list of the predicted output.
        """


        return data[0].tolist()


