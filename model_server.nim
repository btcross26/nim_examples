# A simple server setup to accept a json-POST request of input features for
# the XGBoost model and score the model in response to the request.

# author: Benjamin Cross
# created: 2021-04-07

import os
import asyncdispatch
import asynchttpserver
import times
import strutils
import json


# dynamic library declarations
{.push cdecl, importc, dynlib: "libxgbmodel.dylib".}
proc loadModel()
proc modelPredict(jsonInputs: cstring, score: var cfloat): cstring   # var type works here
proc freeModel()
{.pop.}


# proc for responding to a score request
proc cb(req: Request): Future[void] {.async.} =
  let headers = {
    "Timestamp": $now(),
    "Content-type": "application/json"
  }
  var
    code: HttpCode
    responseHeaders: HttpHeaders = newHttpHeaders(headers)
    outputJson: JsonNode
    score: float32
    message: string
  outputJson = %* {"score": newJNull()}
  if req.reqMethod == HttpPost:
    message = $modelPredict(cstring(req.body), score)
    outputJson["status"] = %message
    if message == "OK":
      outputJson["score"] = %score
      code = Http200
    elif message == "input json error":
      code = Http400
    else:
      code = Http500
  else:
    outputJson["status"] = %"invalid request type"
    code = Http405
  await req.respond(code, pretty(outputJson) & "\c\L", responseHeaders)


# main server function
# note: the acceptRequest proc in version 1.4.2 takes three arguments, the
# second being a Port. This second argument is eliminated in later versions.
# So the below code will not work for 1.4.4. It can be fixed by removing the
# port argument in the acceptRequest call.
proc main(port: Port) {.async.} =
  var
    server = newAsyncHttpServer()
  server.listen(port)
  while true:
    if server.shouldAcceptRequest():
      await server.acceptRequest(port, cb)
    else:
      poll()


when isMainModule:
  var port: Port
  if paramCount() == 0:
    port = Port(8080)
  else:
    port = Port(paramStr(1).parseInt())
  loadModel()
  waitFor main(port)
  freeModel()  # need to add in SIGKILL detection to actually get here
