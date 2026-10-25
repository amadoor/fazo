print(string.format("%d", "10"), string.format("%x", " 255 "), string.format("%5.1f", "2.25"))
print(pcall(string.format, "%d", "x"))
print(pcall(string.format, "%d", "1.5"))
