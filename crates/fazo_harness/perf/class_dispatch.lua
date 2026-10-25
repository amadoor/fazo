local Shape = {}
Shape.__index = Shape
function Shape.new(kind, size)
    return setmetatable({ kind = kind, size = size }, Shape)
end
function Shape:area()
    return self.size * self.size
end
local Circle = setmetatable({}, { __index = Shape })
Circle.__index = Circle
function Circle.new(size)
    return setmetatable({ kind = "circle", size = size }, Circle)
end
function Circle:area()
    return self.size * self.size * 3 // 1
end
local shapes = {}
for i = 1, 200 do
    shapes[i] = (i % 2 == 0) and Shape.new("square", i % 13) or Circle.new(i % 11)
end
local total = 0
for round = 1, 2000 do
    for i = 1, #shapes do
        total = total + shapes[i]:area() % 1009
    end
end
print(total)
