local gen = coroutine.wrap(function()
    for i = 1, 500000 do coroutine.yield(i) end
    return -1
end)
local sum = 0
for _ = 1, 500000 do sum = sum + gen() end
print(sum, gen())
