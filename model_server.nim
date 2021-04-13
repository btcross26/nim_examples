import os
import asyncdispatch
import asyncnet
import asynchttpserver
import times
import strutils
import json

#
# dynamic library definitions
{.push cdecl, importc, dynlib: "libxgbmodel.dylib".}
proc loadModel()
proc modelPredict(jsonInputs: cstring): cfloat
proc freeModel()
{.pop.}
#
# # # error definitions copied from previous files
# # type
# #   InvalidInputJsonError = object of CatchableError
# #   XGBError = object of CatchableError
# #
# # # SIGINT catch (borrowed from https://gist.github.com/dom96/908782)
# # type
# #   TSignal = enum
# #     SIGINT = 2
# # proc signal(sig: cint, `func`: pointer) {.importc: "signal", header: "<signal.h>".}
#
# type
#   InvalidInputJsonError = object of CatchableError
#
proc scoreModel(inputs: string): Future[float32] =
  var
    future: Future[float32] = newFuture[float32]()
    score: float32
  score = float32(modelPredict(cstring(inputs)))
  return future
#
# proc for responding to a score request
proc cb(req: Request): Future[void] =
  raise newException(ValueError, "this is a dummy exception")
  let headers = {
    "Date": $now(),
    "Content-type": "application/json"
  }
  var
    code: HttpCode
    responseHeaders: HttpHeaders = newHttpHeaders(headers)
    outputJson: JsonNode
    score: float32
    futureScore: Future[float32]
  if req.reqMethod == HttpPost:
    futureScore = scoreModel(req.body)
    score = waitFor futureScore
    if futureScore.failed:
      await req.respond(code, "Bad Input", responseHeaders)
      return
    score = future.read()
    outputJson = %* {"score": score, "status": "Success"}
  else:
    outputJson = %* {"score": newJNull(), "status": "$1 not supported" % [$req.reqMethod]}
  await req.respond(code, $outputJson, responseHeaders)
#
# # main server function
# # note: the acceptRequest proc in version 1.4.2 takes three arguments, the
# # second being a Port. This second argument is eliminated in later versions.
# # So the below code will not work for 1.4.4. It can be fixed by removing the
# # port argument in the acceptRequest call.
# proc main(port: Port) {.async.}=
#   var
#     future: Future[void]
#     server = newAsyncHttpServer()
#   server.listen(port)
#   while true:
#     if server.shouldAcceptRequest():
#       future = server.acceptRequest(port, cb)
#       echo "You are here"
#       yield future
#       if future.failed:
#         echo "Future dog!"
#       elif future.finished:
#         echo "Future cat!"
#     else:
#       poll()
#
# when isMainModule:
#   var port: Port
#   if paramCount() == 0:
#     port = Port(8080)
#   else:
#     port = Port(paramStr(1).parseInt())
#   loadModel()
#   waitFor main(port)
#   freeModel()


# proc acceptRequest2(server: AsyncHttpServer,
#   callback: proc(request: Request): Future[void] {.closure, gcsafe.}) {.async.} =
#   ## Accepts a single request. Write an explicit loop around this proc so that
#   ## errors can be handled properly.
#   var (address, client) = await server.socket.acceptAddr()
#   await processClient(server, client, address, callback)


proc main() {.async.} =
  var
    future: Future[void]
    server = newAsyncHttpServer()
  server.listen(Port(8080))
  while true:
    if server.shouldAcceptRequest():
      await server.acceptRequest(Port(8080), cb)
      echo "You are here!"
      yield future
      if future.failed:
        echo "Future dog!"
      elif future.finished:
        echo "Future cat!"
    else:
      poll()

loadModel()
waitFor main()
freeModel()




## Module for handling and raising signals. Wraps ``<signals.h>``.


#
#
# template atSignal*(s: TSignal, actions: stmt): stmt =
#   proc callback(sig: cint) =
#     actions
#   signal(int(s), callback)
#
# when isMainModule:
#   atSignal(SIGINT):
#     echo("Called back!")
#     quit()
#   while True:
#     nil


