local log = {}
local proxy = setmetatable({}, {__newindex = function(t, k, v)
  log[#log + 1] = k
  rawset(t, k, v * 2)
end})
proxy.a = 1
proxy.a = 5
proxy.b = 3
print(proxy.a, proxy.b, #log, log[1], log[2])
local store = {}
local redirect = setmetatable({}, {__newindex = store})
redirect.x = 7
print(rawget(redirect, "x"), store.x)
local readonly = setmetatable({}, {__newindex = function(t, k) error("read only: " .. k, 0) end})
print(pcall(function() readonly.z = 1 end))
