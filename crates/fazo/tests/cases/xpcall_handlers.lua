print(xpcall(function() error("boom") end, function(m) return "handled: " .. m end))
print(xpcall(function(a, b) return a + b end, print, 2, 3))
print(xpcall(error, function(m) error(m) end, "x"))
print(xpcall(error, function(m) error("again") end, "x"))
local n = 0
print(xpcall(error, function(m) n = n + 1 if n == 1 then error("second") end return "H:" .. tostring(m) end, "x"))
print(n)
print(pcall(xpcall, function() end, 1))
print(pcall(xpcall, function() return "no error" end, nil))
print(xpcall(function() error({code = 7}) end, function(e) return e.code end))
print(xpcall(function() error("inner") end, function(m) return (select(2, pcall(error, "in handler"))) .. "|" .. m end))
print(xpcall(function() local function f() return f() + 1 end return f() end,
  function(m) return "overflow handled: " .. tostring(m):match("stack overflow") end))
