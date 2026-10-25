local E = {__eq = function(a, b) return a.id == b.id end}
local a = setmetatable({id = 1}, E)
local b = setmetatable({id = 1}, E)
local c = setmetatable({id = 2}, E)
print(a == b, a == c, a ~= b, a == a, a == 1, a == "1")
local L = {}
L.__lt = function(x, y) return x.n < y.n end
L.__le = function(x, y) return x.n <= y.n end
local p = setmetatable({n = 1}, L)
local q = setmetatable({n = 2}, L)
print(p < q, p <= q, p > q, p >= q, q < p)
if p < q then print("lt branch") end
print(not (p < q))
