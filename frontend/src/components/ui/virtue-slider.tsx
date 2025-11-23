import * as React from "react";
import * as SliderPrimitive from "@radix-ui/react-slider";
import { cn } from "@/lib/utils";

const VirtueSlider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root>
>(({ className, value = [5], onValueChange, ...props }, ref) => {

  // ENSURE 0.00–1.00 VALUES (convert from 0-10 slider)
  const safeValue = Math.min(1, Math.max(0, value[0] / 10));

  const handle = (newVal: number[]) => {
    const v = Math.round((newVal[0] / 10) * 100) / 100; // ← Convert 0-10 slider to 0.00-1.00
    onValueChange?.([v]);
  };

  return (
    <div className={cn("relative flex flex-col items-center", className)}>

      {/* Decimal readout */}
      <div className="h-4 flex items-center justify-center mb-1">
        <div className="text-xs text-gray-600 font-mono select-none">
          {safeValue.toFixed(2)}
        </div>
      </div>

      <div className="flex flex-row items-start">

        {/* Scale */}
        <div className="flex flex-col justify-between text-[10px] text-gray-500 select-none mr-1 h-40">
          {Array.from({ length: 11 }).map((_, i) => (
            <span key={i}>{(1 - i / 10).toFixed(1)}</span>
          ))}
        </div>

        {/* Slider */}
        <SliderPrimitive.Root
          ref={ref}
          value={[safeValue * 10]} // Convert 0.0-1.0 to 0-10 for slider
          onValueChange={handle}
          orientation="vertical"
          min={0}
          max={10}
          step={1}
          className="relative flex flex-col h-40 w-8 touch-none items-center"
          {...props}
        >
          <SliderPrimitive.Track className="relative h-full w-2 bg-gray-300 border border-gray-400 rounded-full overflow-hidden">
            <div
              className="absolute bottom-0 w-full opacity-70"
              style={{
                height: `${safeValue * 100}%`,
                background: "linear-gradient(to top, red, yellow 40%, green 70%)",
              }}
            />
          </SliderPrimitive.Track>

          {/* HIDDEN THUMB */}
          <SliderPrimitive.Thumb className="hidden" />

        </SliderPrimitive.Root>
      </div>
    </div>
  );
});

VirtueSlider.displayName = "VirtueSlider";

export { VirtueSlider };

