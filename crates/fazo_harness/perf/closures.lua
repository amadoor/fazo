local function make(n)
    local count = n
    return function(step)
        count = count + step
        return count
    end
end
local total = 0
for i = 1, 100000 do
    local f = make(i)
    total = total + f(1) + f(2)
end
print(total)
