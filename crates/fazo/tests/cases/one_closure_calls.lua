local function counter(step)
    local total = 0
    return function(n)
        total = total + n * step
        return total
    end
end

local function drive(f, n)
    local sum = 0
    for i = 1, n do
        sum = sum + f(i)
    end
    return sum
end

local first = counter(1)
for _ = 1, 300 do
    drive(first, 20)
end
print(drive(first, 20))
local second = counter(3)
print(drive(second, 20))
print(drive(first, 5), drive(second, 5))

local Class = {}
Class.__index = Class
function Class.new(value)
    return setmetatable({ value = value }, Class)
end
function Class:get(extra)
    if extra > 1000 then
        local made = { self.value }
        return made[1] + extra, debug.traceback("deep", 1):match("traceback") ~= nil
    end
    return self.value + extra
end

local objects = {}
for i = 1, 50 do
    objects[i] = Class.new(i)
end
local total = 0
for round = 1, 200 do
    for i = 1, #objects do
        total = total + objects[i]:get(round)
    end
end
print(total)
print(objects[3]:get(2000))

local wide = Class.new(7)
for i = 1, 40 do
    wide["k" .. i] = i
end
for i = 1, 40 do
    wide["k" .. i] = nil
end
print(wide:get(1), objects[1]:get(1))

local function shared(seed)
    return function(x)
        return x + seed
    end
end
local adders = {}
for i = 1, 4 do
    adders[i] = shared(i)
end
local function apply(f, x)
    return f(x) + f(x + 1)
end
local sum = 0
for round = 1, 400 do
    sum = sum + apply(adders[1], round)
end
for round = 1, 400 do
    sum = sum + apply(adders[round % 4 + 1], round)
end
print(sum)
