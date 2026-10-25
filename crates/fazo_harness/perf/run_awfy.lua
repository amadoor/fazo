local dir = arg[0]:match("^(.*)/") or "."
package.path = dir .. "/awfy/?.lua;" .. package.path
local name, iterations = arg[1], tonumber(arg[2])
local benchmark = require(name)
print(name, benchmark:inner_benchmark_loop(iterations))
