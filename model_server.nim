import os
import asyncdispatch
import asynchttpserver
import times
import strutils
import strformat
import json

# # dynamic library definitions
# {.push cdecl, importc, dynlib: "libxgbmodel.dylib".}
# proc loadModel()
# proc modelPredict(jsonInputs: cstring): cfloat
# proc freeModel()
# {.pop.}
#
# proc cb(req: Request) {.async.} =
#   let headers = {"Date": "Now", "Content-type": "text/plain; charset=utf-8"}
#   await req.respond(Http200, "Hello World", headers.newHttpHeaders())
# proc for responding to a score request
# proc cb(req: Request): Future[void] {.async.} =
#   let headers = {
#     "Date": $now(),
#     "Content-type": "application/json"
#   }
#   var
#     response: JsonNode
#     code: HttpCode
#     responseHeaders: HttpHeaders = newHttpHeaders(headers)
#   if req.reqMethod == HttpPost:
#     echo "Hey"
#     try:
#       echo "I am here"
#       var score: float = modelPredict(cstring(req.body))
#       response = %* {"score": score, "status": "success"}
#       code = Http200
#     except:
#       echo "Now I am here"
#       response = %* {"score": nil, "status": "error"}
#       code = Http422
#   else:
#       response = %* {"score": nil, "status": "expected POST request"}
#       code = Http405
#   await req.respond(code, $response, responseHeaders)

# main server function
# note: the acceptRequest proc in version 1.4.2 takes three arguments, the
# second being a Port. This second argument is eliminated in later versions.
# So the below code will not work for 1.4.4. It can be fixed by removing the
# port argument in the acceptRequest call.
# proc main(port: Port): Future[void] {.async.} =
#   var server: AsyncHttpServer = newAsyncHttpServer()
#   server.listen(port)
#   while true:
#     if server.shouldAcceptRequest():
#       asyncCheck(server.acceptRequest(port, cb))
#     else:
#       poll()
#
#
# when isMainModule:
#   var port: Port
#   if paramCount() == 0:
#     port = Port(8080)
#   else:
#     port = Port(paramStr(1).parseInt())
#   loadModel()
#   waitFor main(port)
#   # runForever()
#   freeModel()

import asynchttpserver, asyncdispatch

proc main {.async.} =
  var server = newAsyncHttpServer()
  proc cb(req: Request) {.async.} =
    let headers = {"Date": "Tue, 29 Apr 2014 23:40:08 GMT",
        "Content-type": "text/plain; charset=utf-8"}
    await req.respond(Http200, "Hello World", headers.newHttpHeaders())

  server.listen Port(8080)
  while true:
    if server.shouldAcceptRequest():
      asyncCheck server.acceptRequest(Port(8080), cb)
    else:
      poll()

asyncCheck main()
runForever()
