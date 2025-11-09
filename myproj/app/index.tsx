// app/index.tsx
import { Ionicons } from "@expo/vector-icons";
import DateTimePicker, { DateTimePickerAndroid } from "@react-native-community/datetimepicker";
import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Image,
  Modal,
  Platform,
  Pressable,
  ScrollView,
  Text,
  TextInput,
  TouchableOpacity,
  TouchableWithoutFeedback,
  UIManager,
  useWindowDimensions,
  View
} from "react-native";
import { Calendar } from "react-native-calendars";
import Icon from "../assets/images/icon4.svg";

/** === Map points (x/y can be 0..1 relative or native px > 1) === */
type Point = { id: string; x: number; y: number; label: string; color?: string };
const MAP_POINTS: Point[] = [
  { id: "Piraeus", x: 1095, y: 1832, label: "Piraeus", color: "#E63946" },
  { id: "Agios Dimitrios", x: 3503, y: 1873, label: "Agios Dimitrios", color: "#E63946" },
  { id: "Exarcheia", x: 2555, y: 343, label: "Exarcheia", color: "#E63946" },
  { id: "Kallithea", x: 2729, y: 1382, label: "Kallithea", color: "#E63946" },
  { id: "Nea Smyrni", x: 2963, y: 1749, label: "Nea Smyrni", color: "#E63946" },
  { id: "Pangrati", x: 3833, y: 967, label: "Pangrati", color: "#E63946" },
  { id: "Syntagma", x: 2587, y: 779, label: "Syntagma", color: "#E63946" },
  { id: "Zografou", x: 4526, y: 651, label: "Zografou", color: "#E63946" }
];

/** === Simple place metadata + alternatives (stub) === */
const PLACE_DETAILS: Record<string, { name: string; image: string; score: number }> = {
  Exarcheia: { name: "Exarcheia", image: require("../assets/images/exarcheia.jpg"), score: 50 },
  Piraeus: { name: "Piraeus", image: require("../assets/images/peireas.jpg"), score: 30 },
  AgiosDimitrios: { name: "Agios Dimitrios", image: require("../assets/images/agios_dimitrios.jpg"), score: 45 },
  // Also accept the id with a space (used in MAP_POINTS) so lookups work regardless
  "Agios Dimitrios": { name: "Agios Dimitrios", image: require("../assets/images/agios_dimitrios.jpg"), score: 45 },
  Kallithea: { name: "Kallithea", image: require("../assets/images/kallithea.jpg"), score: 60 },
  NeaSmyrni: { name: "Nea Smyrni", image: require("../assets/images/nea_smyrni.jpg"), score: 25 },
  // Accept MAP_POINTS id with space as well
  "Nea Smyrni": { name: "Nea Smyrni", image: require("../assets/images/nea_smyrni.jpg"), score: 25 },
  Pangrati: { name: "Pangrati", image: require("../assets/images/pangrati.jpg"), score: 20 },
  Syntagma: { name: "Syntagma", image: require("../assets/images/syntagma.jpg"), score: 85 },
  Zografou: { name: "Zografou", image: require("../assets/images/zografou.jpg"), score: 15 }
};

// Also provide entries that exactly match the `MAP_POINTS` ids (which include spaces)
// so lookups like PLACE_DETAILS[selectedPointId] will work even if keys were stored
// without spaces.
PLACE_DETAILS["Agios Dimitrios"] = PLACE_DETAILS.AgiosDimitrios;
PLACE_DETAILS["Nea Smyrni"] = PLACE_DETAILS.NeaSmyrni;

const ALTERNATIVES: Record<string, string[]> = {
  Exarcheia: ["Zografou","Nea Smyrni"],
  Piraeus: ["Kallithea", "Syntagma"],
  Kallithea: ["Piraeus", "Nea Smyrni", "Zografou"],
  Syntagma: ["Nea Smyrni", "Kallithea"],
  "Agios Dimitrios": ["Nea Smyrni", "Pangrati"],
  "Nea Smyrni": ["Zografou", "Pangrati", "Piraeus",  "Nea Smyrni"],
  Pangrati: ["Agios Dimitrios", "Nea Smyrni"],
  Zografou: ["Nea Smyrni"]
};

/** === Helpers === */
function isoToday(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}
function humanDate(iso: string): string {
  const d = new Date(`${iso}T00:00:00`);
  return d.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short", year: "numeric" });
}
function humanTime(d: Date): string {
  return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}
function scoreLevel(score: number): "Low" | "Mid" | "High" {
  if (score >= 67) return "High";
  if (score >= 34) return "Mid";
  return "Low";
}
function scoreColor(score: number): string {
  const lvl = scoreLevel(score);
  if (lvl === "High") return "#E53935";
  if (lvl === "Mid") return "#FB8C00";
  return "#2E7D32";
}

/** === Map with clickable points === */
type MapImageHandle = { focusPoint: (p: Point) => void };
type MapImageProps = { selectedId: string | null; onPointPress: (id: string) => void };

const MapImage = React.forwardRef<MapImageHandle, MapImageProps>(({ selectedId, onPointPress }, ref) => {
  const mapAsset = require("../assets/images/map.png");
  const { width: winW, height: winH } = useWindowDimensions();
  const [imgSize, setImgSize] = useState<{ width: number; height: number }>({ width: 0, height: 0 });
  const [nativeSize, setNativeSize] = useState<{ width: number; height: number } | null>(null);
  const outerRef = React.useRef<ScrollView | null>(null);
  const innerRef = React.useRef<ScrollView | null>(null);
  const [viewportWidth, setViewportWidth] = useState<number>(0);
  const [viewportHeight, setViewportHeight] = useState<number>(0);

  useEffect(() => {
    const resolved = Image.resolveAssetSource(mapAsset);
    let { width, height } = resolved;
    setNativeSize({ width: Math.round(width), height: Math.round(height) });
    const maxDim = Math.max(winW, winH) * 3;
       const scale = Math.min(1, maxDim / Math.max(width, height));
    width = Math.round(width * scale);
    height = Math.round(height * scale);
    setImgSize({ width, height });
  }, [winW, winH]);

  const displayW = imgSize.width;
  const displayH = imgSize.height;
  const circleSize = 22;
  const labelOffset = 26;

  React.useImperativeHandle(
    ref,
    () => ({
      focusPoint: (p: Point) => {
        let px: number;
        let py: number;
        if (p.x <= 1 && p.y <= 1) {
          px = Math.round(p.x * displayW);
          py = Math.round(p.y * displayH);
        } else if (nativeSize) {
          const scaleX = displayW / nativeSize.width;
          const scaleY = displayH / nativeSize.height;
          px = Math.round(p.x * scaleX);
          py = Math.round(p.y * scaleY);
        } else {
          px = 0;
          py = 0;
        }

        const vw = viewportWidth || 0;
        const vh = viewportHeight || 0;

        const maxScrollX = Math.max(0, displayW - vw);
        const maxScrollY = Math.max(0, displayH - vh);

        const targetX = Math.max(0, Math.min(maxScrollX, px - Math.round(vw / 2)));
        const targetY = Math.max(0, Math.min(maxScrollY, py - Math.round(vh / 2)));

        try {
          outerRef.current?.scrollTo({ x: targetX, animated: true });
        } catch {}
        try {
          innerRef.current?.scrollTo({ y: targetY, animated: true });
        } catch {}
      }
    }),
    [displayW, displayH, nativeSize, viewportWidth, viewportHeight]
  );

  if (imgSize.width === 0 || imgSize.height === 0 || !nativeSize) {
    return (
      <View style={{ flex: 1, backgroundColor: "#e6e6e6", alignItems: "center", justifyContent: "center" }}>
        <Text style={{ color: "#666" }}>Loading map...</Text>
      </View>
    );
  }

  return (
    <View
      style={{ flex: 1 }}
      onLayout={(e) => {
        const { width, height } = e.nativeEvent.layout;
        setViewportWidth(width);
        setViewportHeight(height);
      }}
    >
      <ScrollView ref={outerRef} horizontal style={{ flex: 1 }} contentContainerStyle={{ alignItems: "flex-start" }}>
        <ScrollView ref={innerRef} contentContainerStyle={{ alignItems: "flex-start" }}>
          <View style={{ width: displayW, height: displayH }}>
            <Image source={mapAsset} style={{ width: displayW, height: displayH }} resizeMode="cover" />

            <View style={{ position: "absolute", left: 0, top: 0, width: displayW, height: displayH }} pointerEvents="box-none">
              {MAP_POINTS.map((p) => {
                let px: number;
                let py: number;
                if (p.x <= 1 && p.y <= 1) {
                  px = Math.round(p.x * displayW);
                  py = Math.round(p.y * displayH);
                } else {
                  const scaleX = displayW / nativeSize.width;
                  const scaleY = displayH / nativeSize.height;
                  px = Math.round(p.x * scaleX);
                  py = Math.round(p.y * scaleY);
                }
                const isSelected = selectedId === p.id;
                return (
                  <Pressable
                    key={p.id}
                    onPress={() => onPointPress(p.id)}
                    style={{ position: "absolute", left: px - circleSize / 2, top: py - circleSize / 2, alignItems: "center" }}
                    pointerEvents="box-none"
                  >
                    <View style={{ position: "absolute", bottom: circleSize / 2 + labelOffset, alignItems: "center" }}>
                      <Text
                        style={{
                          backgroundColor: "rgba(255,255,255,0.95)",
                          paddingHorizontal: 6,
                          paddingVertical: 4,
                          borderRadius: 6,
                          overflow: "hidden",
                          width: 160,
                          textAlign: "center",
                          color: "#111",
                          fontSize: 12,
                          lineHeight: 16,
                          borderWidth: 1,
                          borderColor: "#e5e5e5"
                        }}
                        numberOfLines={3}
                      >
                        {p.label}
                      </Text>
                    </View>

                    <View
                      style={{
                        width: circleSize,
                        height: circleSize,
                        borderRadius: circleSize / 2,
                        backgroundColor: isSelected ? "#2ECC71" : p.color ?? "#E63946",
                        borderWidth: 1.5,
                        borderColor: "#fff"
                      }}
                    />
                  </Pressable>
                );
              })}
            </View>
          </View>
        </ScrollView>
      </ScrollView>
    </View>
  );
});
MapImage.displayName = "MapImage";

/** === Score bar === */
function ScoreBar({ score }: { score: number }) {
  return (
    <View style={{ marginTop: 6 }}>
      <View style={{ height: 8, borderRadius: 6, backgroundColor: "#eee", overflow: "hidden" }}>
        <View style={{ height: 8, width: `${Math.max(0, Math.min(100, score))}%`, backgroundColor: scoreColor(score) }} />
      </View>
    </View>
  );
}

/** === Cards === */
function SelectedPlaceCard({ id, onLayout }: { id: string; onLayout?: (h: number) => void }) {
  const details = PLACE_DETAILS[id];
  if (!details) return null;
  const lvl = scoreLevel(details.score);
  const [imgSrc, setImgSrc] = useState<any>(typeof details.image === "string" ? { uri: details.image } : details.image);
  useEffect(() => {
    setImgSrc(typeof details.image === "string" ? { uri: details.image } : details.image);
  }, [details.image]);
  return (
    <View
      key={id}
      onLayout={(e) => onLayout?.(e.nativeEvent.layout.height)}
      style={{
        flex: 1,
        backgroundColor: "#fff",
        borderRadius: 16,
        overflow: "hidden",
        borderWidth: 1,
        borderColor: "#e5e5e5",
        shadowColor: "#000",
        shadowOpacity: 0.1,
        shadowRadius: 10,
        shadowOffset: { width: 0, height: 4 },
        elevation: 6
      }}
    >
      <Image
        source={imgSrc}
        onError={() => setImgSrc({ uri: "https://via.placeholder.com/600x400?text=No+Image" })}
        style={{ width: "100%", height: 140 }}
      />
      <View style={{ padding: 12 }}>
        <Text style={{ fontSize: 18, fontWeight: "800", marginBottom: 6 }}>{details.name}</Text>
        <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between" }}>
          <Text style={{ fontSize: 12, color: "#444" }}>Overtourism Score</Text>
          <Text style={{ fontSize: 12, fontWeight: "800", color: scoreColor(details.score) }}>{lvl}</Text>
        </View>
        <ScoreBar score={details.score} />
      </View>
    </View>
  );
}

function AlternativeCard({
  id,
  onPrev,
  onNext,
  onLayout
}: {
  id: string;
  onPrev: () => void;
  onNext: () => void;
  onLayout?: (h: number) => void;
}) {
  const details = PLACE_DETAILS[id];
  if (!details) return null;
  const lvl = scoreLevel(details.score);
  const [imgSrc, setImgSrc] = useState<any>(typeof details.image === "string" ? { uri: details.image } : details.image);
  useEffect(() => {
    setImgSrc(typeof details.image === "string" ? { uri: details.image } : details.image);
  }, [details.image]);
  return (
    <View
      key={id}
      onLayout={(e) => onLayout?.(e.nativeEvent.layout.height)}
      style={{
        flex: 1,
        backgroundColor: "#fff",
        borderRadius: 16,
        overflow: "hidden",
        borderWidth: 1,
        borderColor: "#e5e5e5",
        shadowColor: "#000",
        shadowOpacity: 0.1,
        shadowRadius: 10,
        shadowOffset: { width: 0, height: 4 },
        elevation: 6
      }}
    >
      <View>
        <Image
          source={imgSrc}
          onError={() => setImgSrc({ uri: "https://via.placeholder.com/600x400?text=No+Image" })}
          style={{ width: "100%", height: 140 }}
        />
        <View
          style={{
            position: "absolute",
            top: 8,
            left: 8,
            right: 8,
            backgroundColor: "rgba(17,17,17,0.7)",
            paddingHorizontal: 8,
            paddingVertical: 4,
            borderRadius: 8
          }}
        >
          <Text style={{ color: "#fff", fontSize: 12, fontWeight: "700" }}>Alternative destination you can visit</Text>
        </View>
      </View>

      <View style={{ padding: 12, flex: 1 }}>
        <Text style={{ fontSize: 16, fontWeight: "800", marginBottom: 6 }}>{details.name}</Text>
        <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between" }}>
          <Text style={{ fontSize: 12, color: "#444" }}>Overtourism Score</Text>
          <Text style={{ fontSize: 12, fontWeight: "800", color: scoreColor(details.score) }}>{lvl}</Text>
        </View>
        <ScoreBar score={details.score} />

        <View style={{ flexDirection: "row", justifyContent: "space-between", marginTop: 12 }}>
          <TouchableOpacity
            onPress={onPrev}
            style={{
              paddingHorizontal: 10,
              paddingVertical: 6,
              borderRadius: 8,
              borderWidth: 1,
              borderColor: "#ddd",
              backgroundColor: "#fafafa"
            }}
          >
            <View style={{ flexDirection: "row", alignItems: "center" }}>
              <Ionicons name="chevron-back" size={14} color="#333" />
              <Text style={{ marginLeft: 4, fontWeight: "700", fontSize: 13 }}>Prev</Text>
            </View>
          </TouchableOpacity>

          <TouchableOpacity
            onPress={onNext}
            style={{
              paddingHorizontal: 10,
              paddingVertical: 6,
              borderRadius: 8,
              borderWidth: 1,
              borderColor: "#ddd",
              backgroundColor: "#fafafa"
            }}
          >
            <View style={{ flexDirection: "row", alignItems: "center" }}>
              <Text style={{ marginRight: 4, fontWeight: "700", fontSize: 13 }}>Next</Text>
              <Ionicons name="chevron-forward" size={14} color="#333" />
            </View>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
}

/** === Screen === */
export default function HomeScreen() {
  useEffect(() => {
    if (Platform.OS === "android" && UIManager.setLayoutAnimationEnabledExperimental) {
      UIManager.setLayoutAnimationEnabledExperimental(true);
    }
  }, []);

  // Date-time
  const [selectedDate, setSelectedDate] = useState<string>(isoToday());
  const [selectedTime, setSelectedTime] = useState<Date>(new Date());
  const [calendarVisible, setCalendarVisible] = useState<boolean>(false);

  // Default selection = first point
  const defaultId = MAP_POINTS[0]?.id ?? null;
  const [selectedPointId, setSelectedPointId] = useState<string | null>(defaultId);

  // Map focus on mount and on selection
  const mapRef = useRef<MapImageHandle | null>(null);
  useEffect(() => {
    if (!selectedPointId) return;
    const p = MAP_POINTS.find((m) => m.id === selectedPointId);
    if (p && mapRef.current?.focusPoint) mapRef.current.focusPoint(p);
  }, [selectedPointId]);

  // Alternatives
  const altIds = useMemo(() => {
    if (!selectedPointId) return MAP_POINTS.map((p) => p.id);
    return ALTERNATIVES[selectedPointId] ?? MAP_POINTS.map((p) => p.id).filter((id) => id !== selectedPointId);
  }, [selectedPointId]);
  const [altIndex, setAltIndex] = useState(0);
  useEffect(() => setAltIndex(0), [selectedPointId]);
  const currentAltId = altIds.length ? altIds[altIndex % altIds.length] : null;

  // Equal height cards; increase base height to make both a bit taller
  const BASE_CARD_MIN = 280;
  const [selH, setSelH] = useState<number>(BASE_CARD_MIN);
  const [altH, setAltH] = useState<number>(BASE_CARD_MIN);
  const targetH = Math.max(selH, altH, BASE_CARD_MIN);

  useEffect(() => {
    setSelH(BASE_CARD_MIN);
    setAltH(BASE_CARD_MIN);
  }, [selectedPointId, currentAltId]);

  const openAndroidTimePicker = () => {
    DateTimePickerAndroid.open({
      value: selectedTime,
      mode: "time",
      is24Hour: true,
      onChange: (_evt, date) => {
        if (date) setSelectedTime(date);
      }
    });
  };

  return (
    <View style={{ flex: 1, backgroundColor: "#fff", paddingTop: 50 }}>
      {/* NAVBAR CARD WITH THIN BORDER + SHADOW */}
      <View style={{ paddingHorizontal: 16, marginBottom: 14 }}>
        <View
          style={{
            flexDirection: "row",
            alignItems: "center",
            justifyContent: "space-between",
            paddingHorizontal: 12,
            paddingVertical: 10,
            backgroundColor: "#fff",
            borderRadius: 16,
            borderWidth: 1,
            borderColor: "#e5e5e5",
            shadowColor: "#000",
            shadowOpacity: 0.12,
            shadowRadius: 12,
            shadowOffset: { width: 0, height: 6 },
            elevation: 8
          }}
        >
          <View style={{ flexDirection: "row", alignItems: "center" }}>
            <Icon width={35} height={35} />
          </View>

          <View
            style={{
              flex: 1,
              flexDirection: "row",
              alignItems: "center",
              backgroundColor: "#f2f2f2",
              borderRadius: 12,
              paddingHorizontal: 10,
              marginHorizontal: 10,
              height: 38,
              borderWidth: 1,
              borderColor: "#eaeaea"
            }}
          >
            <Ionicons name="search" size={18} color="#999" style={{ marginRight: 6 }} />
            <TextInput
              placeholder="Destination"
              placeholderTextColor="#999"
              // Keep it single-line and vertically centered on Android so the text
              // doesn't shift/scroll inside the small bar.
              multiline={false}
              textAlignVertical={Platform.OS === "android" ? "center" : undefined}
              style={{ flex: 1, fontSize: 16, paddingVertical: 0 }}
              underlineColorAndroid="transparent"
            />
          </View>

          <View style={{ flexDirection: "row", alignItems: "center" }}>
            <TouchableOpacity style={{ marginHorizontal: 6 }}>
              <Ionicons name="notifications-outline" size={24} color="#333" />
            </TouchableOpacity>
            <TouchableOpacity style={{ marginHorizontal: 6 }}>
              <Ionicons name="person-circle-outline" size={28} color="#333" />
            </TouchableOpacity>
          </View>
        </View>
      </View>

      {/* MAP CARD SMALLER + EXTRA MARGINS + THIN BORDER + SHADOW */}
      <View style={{ marginTop: 12, marginBottom: 24, paddingHorizontal: 16 }}>
        <View
          style={{
            height: 300,
            borderRadius: 16,
            backgroundColor: "#fff",
            shadowColor: "#000",
            shadowOpacity: 0.12,
            shadowRadius: 12,
            shadowOffset: { width: 0, height: 6 },
            elevation: 10
          }}
        >
          <View
            style={{
              flex: 1,
              borderRadius: 16,
              overflow: "hidden",
              borderWidth: 1,
              borderColor: "#e5e5e5"
            }}
          >
            {/* Floating date-time button */}
            <View style={{ position: "absolute", top: 12, right: 12, zIndex: 2, elevation: 2 }} pointerEvents="box-none">
              <Pressable
                onPress={() => setCalendarVisible(true)}
                style={{
                  flexDirection: "row",
                  alignItems: "center",
                  paddingHorizontal: 10,
                  paddingVertical: 8,
                  borderRadius: 12,
                  backgroundColor: "rgba(255,255,255,0.95)",
                  borderWidth: 1,
                  borderColor: "#ddd",
                  shadowColor: "#000",
                  shadowOpacity: 0.15,
                  shadowRadius: 8,
                  shadowOffset: { width: 0, height: 2 },
                  elevation: 3
                }}
              >
                <Ionicons name="calendar-outline" size={16} color="#333" />
                <Text style={{ marginLeft: 6, fontSize: 14, fontWeight: "600", color: "#333" }}>
                  {humanDate(selectedDate)} • {humanTime(selectedTime)}
                </Text>
              </Pressable>
            </View>

            <MapImage ref={mapRef} selectedId={selectedPointId} onPointPress={(id) => setSelectedPointId(id)} />
          </View>
        </View>
      </View>

      {/* BOTTOM CARDS: FILL REMAINING SPACE, EQUAL HEIGHT */}
      <View style={{ flex: 1, paddingHorizontal: 16, paddingBottom: 16 }}>
        <View style={{ flexDirection: "row", flex: 1 }}>
          {/* Left: Picked place */}
          <View style={{ flex: 1, marginRight: 8 }}>
            {selectedPointId ? (
              <View style={{ height: targetH }}>
                <SelectedPlaceCard id={selectedPointId} onLayout={(h) => setSelH(h)} />
              </View>
            ) : (
              <View
                style={{
                  height: targetH,
                  backgroundColor: "#fff",
                  borderRadius: 16,
                  borderWidth: 1,
                  borderColor: "#e5e5e5",
                  alignItems: "center",
                  justifyContent: "center",
                  shadowColor: "#000",
                  shadowOpacity: 0.06,
                  shadowRadius: 8,
                  shadowOffset: { width: 0, height: 3 },
                  elevation: 3
                }}
              >
                <Text style={{ color: "#666" }}>Tap a point on the map</Text>
              </View>
            )}
          </View>

          {/* Right: Alternative */}
          <View style={{ flex: 1, marginLeft: 8 }}>
            {currentAltId ? (
              <View style={{ height: targetH }}>
                <AlternativeCard
                  id={currentAltId}
                  onPrev={() => setAltIndex((i) => (i - 1 + altIds.length) % altIds.length)}
                  onNext={() => setAltIndex((i) => (i + 1) % altIds.length)}
                  onLayout={(h) => setAltH(h)}
                />
              </View>
            ) : (
              <View
                style={{
                  height: targetH,
                  backgroundColor: "#fff",
                  borderRadius: 16,
                  borderWidth: 1,
                  borderColor: "#e5e5e5",
                  alignItems: "center",
                  justifyContent: "center",
                  shadowColor: "#000",
                  shadowOpacity: 0.06,
                  shadowRadius: 8,
                  shadowOffset: { width: 0, height: 3 },
                  elevation: 3
                }}
              >
                <Text style={{ color: "#666" }}>No alternatives</Text>
              </View>
            )}
          </View>
        </View>
      </View>

      {/* Modal: date + time */}
      <Modal visible={calendarVisible} transparent animationType="fade" onRequestClose={() => setCalendarVisible(false)}>
        <TouchableWithoutFeedback onPress={() => setCalendarVisible(false)}>
          <View style={{ flex: 1, backgroundColor: "rgba(0,0,0,0.35)", justifyContent: "center", alignItems: "center", padding: 16 }}>
            <TouchableWithoutFeedback>
              <View
                style={{
                  width: "90%",
                  maxWidth: 420,
                  backgroundColor: "#fff",
                  borderRadius: 16,
                  borderWidth: 1,
                  borderColor: "#ddd",
                  overflow: "hidden"
                }}
              >
                <View
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    justifyContent: "space-between",
                    paddingHorizontal: 12,
                    paddingVertical: 10,
                    borderBottomWidth: 1,
                    borderBottomColor: "#eee"
                  }}
                >
                  <Text style={{ fontSize: 16, fontWeight: "700" }}>Select date & time</Text>
                  <TouchableOpacity onPress={() => setCalendarVisible(false)} hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}>
                    <Ionicons name="close" size={20} color="#333" />
                  </TouchableOpacity>
                </View>

                <Calendar
                  initialDate={selectedDate}
                  onDayPress={(day: { dateString: string }) => setSelectedDate(day.dateString)}
                  markedDates={{ [selectedDate]: { selected: true } }}
                  enableSwipeMonths
                  theme={{
                    todayTextColor: "#E63946",
                    selectedDayBackgroundColor: "#111",
                    selectedDayTextColor: "#fff",
                    arrowColor: "#111",
                    textMonthFontWeight: "700"
                  }}
                />

                <View style={{ paddingHorizontal: 12, paddingVertical: 12, borderTopWidth: 1, borderTopColor: "#eee" }}>
                  <Text style={{ fontSize: 14, fontWeight: "700", marginBottom: 8 }}>Time</Text>

                  {Platform.OS === "ios" ? (
                    <DateTimePicker
                      mode="time"
                      display="spinner"
                      value={selectedTime}
                      onChange={(_e, d) => d && setSelectedTime(d)}
                      style={{ height: 160 }}
                    />
                  ) : (
                    <TouchableOpacity
                      onPress={openAndroidTimePicker}
                      style={{
                        alignSelf: "flex-start",
                        flexDirection: "row",
                        alignItems: "center",
                        paddingHorizontal: 10,
                        paddingVertical: 8,
                        borderRadius: 10,
                        borderWidth: 1,
                        borderColor: "#ddd",
                        backgroundColor: "#fafafa"
                      }}
                    >
                      <Ionicons name="time-outline" size={16} color="#333" />
                      <Text style={{ marginLeft: 6, fontSize: 14, fontWeight: "600", color: "#333" }}>{humanTime(selectedTime)}</Text>
                      <Ionicons name="chevron-down" size={16} color="#333" style={{ marginLeft: 4 }} />
                    </TouchableOpacity>
                  )}
                </View>

                <View style={{ flexDirection: "row", justifyContent: "flex-end", paddingHorizontal: 12, paddingVertical: 12 }}>
                  <TouchableOpacity
                    onPress={() => setCalendarVisible(false)}
                    style={{
                      paddingHorizontal: 14,
                      paddingVertical: 10,
                      borderRadius: 10,
                      borderWidth: 1,
                      borderColor: "#ddd",
                      backgroundColor: "#fff"
                    }}
                  >
                    <Text style={{ fontWeight: "600" }}>Cancel</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    onPress={() => setCalendarVisible(false)}
                    style={{
                      paddingHorizontal: 14,
                      paddingVertical: 10,
                      borderRadius: 10,
                      backgroundColor: "#111",
                      marginLeft: 8
                    }}
                  >
                    <Text style={{ color: "#fff", fontWeight: "700" }}>Done</Text>
                  </TouchableOpacity>
                </View>
              </View>
            </TouchableWithoutFeedback>
          </View>
        </TouchableWithoutFeedback>
      </Modal>
    </View>
  );
}
