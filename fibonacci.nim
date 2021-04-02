import nimpy

proc fib_recursive(n: int) : int {.exportpy} =
  ## Calculate the `n`th fibonacci number.
  ##
  ## The standard (bad) recursive fibonacci implementation for comparison.
  ##
  ## Parameters
  ## ----------
  ## n: int
  ##   The index of the fibonacci number calculate.
  if n == 0:
    return 0
  elif n == 1:
    return 1
  else:
    return fib_recursive(n - 2) + fib_recursive(n - 1)


proc fib_loop(n: int): int {.exportpy.} =
  ## Calculate the `n`th fibonacci number.
  ##
  ## A more efficient looping (non-recursive) fibonacci implementation.
  ##
  ## Parameters
  ## ----------
  ## n: int
  ##   The index of the fibonacci number calculate.
  if n == 0:
    return 0
  elif n == 1:
    return 1

  var
    last: int = 0
    current: int = 1
    value: int

  for i in 2..<n:
    value = last + current
    last = current
    current = value
  return value
