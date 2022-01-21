#!/bin/bash
# curl http://127.0.0.1:8080/predictions/energy_predictor -T molecule_request.json


cat > instances.json <<END
{
  "instances": [
    {
      "data": {
        "b64": "$(base64 --wrap=0 ./molecule_request.json)"
      }
    }
  ]
}
END


curl -X POST -H "Content-Type: application/json; charset=utf-8" -d @instances.json localhost:8080/predictions/energy_predictor
