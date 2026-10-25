local co = coroutine.create(function()
  local ok, a = pcall(function()
    local x = coroutine.yield("in pcall")
    error("after yield " .. x)
  end)
  coroutine.yield(ok, a)
  return "finished"
end)
print(coroutine.resume(co))
print(coroutine.resume(co, "value"))
print(coroutine.resume(co))
print(coroutine.status(co))

local mt = {__index = function(t, k) return coroutine.yield("index " .. k) end,
             __add = function(a, b) return coroutine.yield("add") end}
local co2 = coroutine.wrap(function()
  local t = setmetatable({}, mt)
  local v = t.name
  local s = t + 1
  return v, s
end)
print(co2())
print(co2("resolved"))
print(co2("sum"))

local co3 = coroutine.wrap(function()
  print(pcall(table.sort, {1, 2, 3}, coroutine.yield))
  print(pcall(string.gsub, "abc", "b", coroutine.yield))
  return "end"
end)
print(co3())
print(pcall(co3))
print(pcall(coroutine.yield))

local co4 = coroutine.wrap(function(...)
  local args = {...}
  local got = {coroutine.yield(#args)}
  return table.unpack(got)
end)
print(co4(1, 2, 3))
print(co4("a", nil, "c"))
print(select('#', coroutine.wrap(function() return end)()))
