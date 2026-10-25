local co = coroutine.create(function(a)
  local b = coroutine.yield(a + 1)
  print("inside", b, coroutine.status(coroutine.running()))
  local c = coroutine.yield(b * 2)
  return c .. "!"
end)
print(coroutine.resume(co, 1))
print(coroutine.resume(co, 10))
print(coroutine.resume(co, "done"))
print(coroutine.status(co), coroutine.resume(co))
local gen = coroutine.wrap(function()
  for i = 1, 4 do coroutine.yield(i * i) end
end)
local acc = {}
for _ = 1, 4 do acc[#acc + 1] = gen() end
print(table.concat(acc, " "))
