local gen = coroutine.wrap(function(limit)
  for i = 1, limit do
    local s = 0
    for j = 1, 20 do s = s + j end
    coroutine.yield(i + s)
  end
  error("gen failed")
end)

local outer = coroutine.create(function()
  print(pcall(function()
    for k = 1, 10 do print("gen", gen(3)) end
  end))
  coroutine.yield("paused outer")
  local ok, err = coroutine.resume(coroutine.create(function() error({code = 9}) end))
  print("nested", ok, type(err), err.code)
  return "outer done"
end)

print(coroutine.resume(outer))
print(coroutine.resume(outer))
print(coroutine.status(outer))
