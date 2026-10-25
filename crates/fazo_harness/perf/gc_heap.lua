collectgarbage("generational")

local HEAP = 200000
local FRAMES = 200
local PER_FRAME = 1500

local heap = {}
for i = 1, HEAP do
    heap[i] = { id = i, weight = i % 97, next = nil }
end
for i = 1, HEAP do
    heap[i].next = heap[(i * 7) % HEAP + 1]
end

local checksum = 0
for frame = 1, FRAMES do
    local scratch = {}
    for j = 1, PER_FRAME do
        local short = { x = j, y = frame, tag = j % 5 }
        scratch[j] = short
        checksum = (checksum + short.x * 3 + short.y + short.tag) % 1000000007
    end
    local slot = (frame * 7919) % HEAP + 1
    heap[slot] = { id = slot, weight = frame, next = heap[(slot * 3) % HEAP + 1] }
    checksum = (checksum + scratch[PER_FRAME].x) % 1000000007
    collectgarbage("step")
end

local total = 0
for i = 1, HEAP do
    total = (total + heap[i].weight + heap[i].next.id) % 1000000007
end
print(checksum, total)
