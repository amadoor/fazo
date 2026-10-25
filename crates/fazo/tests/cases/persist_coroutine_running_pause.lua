local function inner(n)
  local acc = 0
  for i = 1, n do acc = acc + i end
  local echoed, extra = coroutine.yield("inner", acc)
  for i = 1, 25 do acc = acc + i end
  return acc * 2, echoed, extra
end

local function middle()
  local co = coroutine.create(inner)
  local ok, tag, value = coroutine.resume(co, 50)
  print("middle", ok, tag, value, coroutine.status(co))
  local total = 0
  for i = 1, 30 do total = total + i end
  print("middle total", total, coroutine.status(co))
  local ok2, doubled, echoed, extra = coroutine.resume(co, "e", 7)
  print("middle again", ok2, doubled, echoed, extra, coroutine.status(co))
  return total, doubled
end

local outer = coroutine.create(function()
  local a, b = middle()
  print("outer", a, b, coroutine.status(coroutine.running()))
  coroutine.yield("from outer")
  print("outer resumed", coroutine.isyieldable())
  return "done"
end)

print(coroutine.resume(outer))
print(coroutine.resume(outer))
print(coroutine.status(outer))
