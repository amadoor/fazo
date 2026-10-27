local function accumulate(t, n)
    local a = 0
    for i = 1, n do
        a = a + t[i] * 0.5
    end
    return a, math.type(a)
end

local floats = {}
for i = 1, 100 do
    floats[i] = i + 0.25
end
for _ = 1, 200 do
    accumulate(floats, 100)
end
print(accumulate(floats, 100))
print(accumulate(floats, 0))
print(accumulate({ "2", 3.5 }, 2))

local watcher = setmetatable({}, {
    __mul = function()
        local name, value = debug.getlocal(2, 3)
        print(name, value, math.type(value))
        return 1.5
    end,
})
print(accumulate({ watcher, 2.5, watcher }, 3))

local function first_exit(t, n)
    local a = -7
    for i = 1, n do
        a = a + t[i] * 0.5
    end
    return a
end
for _ = 1, 200 do
    first_exit(floats, 100)
end
print(pcall(first_exit, { {} }, 1))
print(pcall(first_exit, { 1.5, {} }, 2))

local function big(t, n)
    local a = math.maxinteger
    for i = 1, n do
        a = a + t[i] * 0.5
    end
    return a, math.type(a)
end
for _ = 1, 200 do
    big(floats, 10)
end
print(big(floats, 10))
print(big(floats, 0))
