print(pcall(string.format, "%.3a", 1.0))
print(pcall(string.format, "%10.2A|%-12a|%+a|% a", 255.5, 0.5, 3.0, 3.0))
print(pcall(string.format, "%#.0a|%012a|%.1a|%.0a", 1.0, -2.0, 1.96875, 1.5))
print(pcall(string.format, "%.20a|%a", 0.1, 5e-324))
