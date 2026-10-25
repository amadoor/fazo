local function three() return 1, 2, 3 end
print(three())
print((three()))
print(three(), 10)
local t = {three(), three()}
print(#t, t[4])
print(({three()})[3])
local function tail(n) if n == 0 then return "done", n end return tail(n - 1) end
print(tail(100000))
local function sum(n, acc) if n == 0 then return acc end return sum(n - 1, acc + n) end
print(sum(200000, 0))
print(string.format("%d %d", three()))
local function none() end
print(none(), select("#", none()))
